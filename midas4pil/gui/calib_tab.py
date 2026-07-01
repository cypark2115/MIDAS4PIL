# Copyright (c) 2026, UChicago Argonne, LLC. All Rights Reserved.
# Author: Changyong Park, HPCAT, X-ray Science Division, Argonne National Laboratory.
# BSD Open Source License -- see LICENSE in the project root for the full license text.
"""
Calibration tab — detector geometry determination.

Layout
------
Left panel (QScrollArea, 300 px)  — QFormLayout-style controls:
  Detector | Beam | Geometry (collapsible) | Calibrant | Actions | Results

Right panel — toggle display:
  Header: [Raw][Cake][Lineout][Mask ●] | position label
  Stacked canvas:
    Raw    — full-canvas TIFF + HistClimWidget side panel
    Cake   — full-canvas 2D cake + HistClimWidget side panel
    Lineout — 1D plot (full width)

Mouse (Raw + Cake):
  Scroll up/down  — zoom in/out centred on cursor (clamped at image bounds)
  Left drag       — rubber-band square → zoom in on release
  Right-click     — undo last zoom
"""

from pathlib import Path
import logging

import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                                QFormLayout, QGroupBox, QComboBox,
                                QPushButton, QLabel, QLineEdit,
                                QCheckBox, QFileDialog, QMessageBox,
                                QScrollArea, QStackedWidget, QSizePolicy,
                                QDialog, QTableWidget, QTableWidgetItem,
                                QHeaderView, QAbstractItemView)
from PySide6.QtCore import Qt, QThread, Signal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


class _NavBar(NavigationToolbar2QT):
    """Toolbar with layout/property items removed — not applicable to axes
    created via fig.add_axes() with explicit coordinates."""
    toolitems = [t for t in NavigationToolbar2QT.toolitems
                 if t[0] not in ('Subplots', 'Customize')]

from ..io import (make_geometry, save_params, load_tiff, auto_mask, load_mask,
                  read_poni, write_poni)
from ..geometry import (build_lut, find_beam_center,
                        find_beam_center_sharpness, find_beam_center_ellipse,
                        lut_tth_range)
from ..integrate import integrate_1d, integrate_1d_varbin
from ..calibrant import load_calibrant
from ..optimizer import calibrate, _CalibrationCancelled
from .detectors import DETECTORS, DETECTOR_NAMES, DEFAULT_DETECTOR, make_panel_map
from .hist_clim import HistClimWidget

log = logging.getLogger(__name__)

_HC     = 12398.4193   # eV·Å
_XH     = '#00e676'    # green crosshair / sync — visible on all themes
_XH2    = '#ff9800'    # orange — ring arc overlay
_FG     = '#333333'    # system theme defaults (overridden by apply_mpl_theme)
_BG_FIG = '#eeeeee'
_BG_AX  = '#f8f8f8'

# Number of pixel widths used as the 2θ acceptance half-window when assigning
# pixels to calibrant rings for both the optimizer and the strain display overlay.
# Dimensionless — independent of detector geometry or beamline.
_TTH_TOL_FACTOR = 3.0

# Bin counts for the caked (2θ, η) image shown in the Calibration tab display.
_N_TTH_DEFAULT = 500
_N_ETA_DEFAULT = 360


def _style_ax(ax, fg=_FG, bg=_BG_AX, spine='#999999'):
    ax.set_facecolor(bg)
    ax.tick_params(colors=fg, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(spine)
    ax.xaxis.label.set_color(fg)
    ax.yaxis.label.set_color(fg)
    ax.title.set_color(fg)


def _refresh_navbar_icons(toolbar):
    """Re-render NavigationToolbar2QT icons with the current palette foreground.

    matplotlib bakes the icon pixmap colour at construction time; this
    re-calls _icon() for each action so icons stay visible after a
    palette / theme switch.
    """
    try:
        for (_text, _tip, image_file, _cb), action in zip(
                toolbar.toolitems, toolbar.actions()):
            if image_file:
                action.setIcon(toolbar._icon(image_file + '.png'))
    except Exception:
        pass


def _list_calibrants():
    cal_dir = Path(__file__).resolve().parent.parent / 'calibrants'
    return sorted(f.stem for f in cal_dir.rglob('*.jcpds'))


def _compute_cake(image, mask, tth_lut, eta_lut, n_tth=_N_TTH_DEFAULT, n_eta=_N_ETA_DEFAULT):
    """2D mean-intensity cake from precomputed LUTs."""
    good = (mask == 0) if mask is not None else np.ones(image.shape, dtype=bool)
    good &= np.isfinite(image)
    if not good.any():
        return None, None, None

    tth_f = tth_lut[good].ravel()
    eta_f = eta_lut[good].ravel()
    I_f   = image[good].ravel()

    t0, t1 = float(np.nanmin(tth_f)), float(np.nanmax(tth_f))
    e0, e1 = -180.0, 180.0

    weights, te, ee = np.histogram2d(
        tth_f, eta_f, bins=[n_tth, n_eta],
        range=[[t0, t1], [e0, e1]], weights=I_f)
    counts, _, _ = np.histogram2d(
        tth_f, eta_f, bins=[n_tth, n_eta],
        range=[[t0, t1], [e0, e1]])

    with np.errstate(divide='ignore', invalid='ignore'):
        cake = np.where(counts > 0, weights / counts, np.nan)

    tth_c = 0.5 * (te[:-1] + te[1:])
    eta_c = 0.5 * (ee[:-1] + ee[1:])
    return cake.T, tth_c, eta_c, counts.T   # (n_eta, n_tth) for imshow


def _compute_cake_mask_rgba(mask, tth_lut, eta_lut, tth_c, eta_c):
    """Project pixel-space mask onto 2θ-η cake coordinates.

    Returns (rgba, extent) where rgba is (n_eta, n_tth, 4) float32 and
    extent is the outer-bin-edge [tth_l, tth_r, eta_b, eta_t], or
    (None, None) when no valid masked pixels are found.
    """
    if mask is None:
        return None, None
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None, None
    tth_vals = tth_lut[rows, cols]
    eta_vals = eta_lut[rows, cols]
    valid    = np.isfinite(tth_vals) & np.isfinite(eta_vals)
    tth_vals = tth_vals[valid]
    eta_vals = eta_vals[valid]
    if len(tth_vals) == 0:
        return None, None

    n_tth, n_eta = len(tth_c), len(eta_c)
    tth_l = float(tth_c[0]  - 0.5*(tth_c[1]  - tth_c[0])  if n_tth > 1 else tth_c[0])
    tth_r = float(tth_c[-1] + 0.5*(tth_c[-1] - tth_c[-2]) if n_tth > 1 else tth_c[-1])
    eta_b = float(eta_c[0]  - 0.5*(eta_c[1]  - eta_c[0])  if n_eta > 1 else eta_c[0])
    eta_t = float(eta_c[-1] + 0.5*(eta_c[-1] - eta_c[-2]) if n_eta > 1 else eta_c[-1])

    ti = np.clip(
        np.floor((tth_vals - tth_l) / (tth_r - tth_l) * n_tth
                 ).astype(int), 0, n_tth - 1)
    ei = np.clip(
        np.floor((eta_vals - eta_b) / (eta_t - eta_b) * n_eta
                 ).astype(int), 0, n_eta - 1)

    cake_mask = np.zeros((n_tth, n_eta), dtype=bool)
    cake_mask[ti, ei] = True
    # cake is displayed as (n_eta, n_tth) with origin='lower'
    rgba = np.zeros((n_eta, n_tth, 4), dtype=np.float32)
    rgba[..., 0] = 1.0
    rgba[..., 3] = 0.65 * cake_mask.T.astype(np.float32)
    return rgba, [tth_l, tth_r, eta_b, eta_t]


# ── Worker ────────────────────────────────────────────────────────────────────

class _FindCenterWorker(QThread):
    """Background worker for Find Center / Auto Find Center."""
    finished = Signal(float, float, float, float, float)  # bc_y, bc_z, lsd, ty_deg, tz_deg
    error    = Signal(str)

    def __init__(self, image, mask, rings, geom, auto=False,
                 rough_max_deg=None, parent=None):
        super().__init__(parent)
        self.image        = image
        self.mask         = mask
        self.rings        = rings
        self.geom         = geom
        self.auto         = auto
        self.rough_max_deg = rough_max_deg

    def run(self):
        import traceback
        try:
            mask = (self.mask if self.mask is not None
                    else np.zeros(self.image.shape, dtype=bool))
            ty_deg = self.geom.get('ty_deg', 0.0)
            tz_deg = self.geom.get('tz_deg', 0.0)

            if self.auto:
                px  = self.geom['px']
                lsd = self.geom['lsd']

                if self.rings:
                    # Rings loaded → full-geometry ring-score finder.
                    # Correctly handles elliptical rings from large tilts by
                    # using pixel_to_r_eta() for the 2θ mapping.  Returns all
                    # five geometry parameters directly.
                    rho_d = self.geom.get('rho_d', 217578.0)
                    bc_y, bc_z, lsd, ty_deg, tz_deg = find_beam_center_ellipse(
                        self.image, mask, self.rings,
                        px=px, rho_d=rho_d,
                        bc_y_init=self.geom.get('bc_y'),
                        bc_z_init=self.geom.get('bc_z'),
                        lsd_init=lsd,
                        ty_init=ty_deg, tz_init=tz_deg)
                    log.info(
                        "Ring-score finder → bc_y=%.1f  bc_z=%.1f  "
                        "lsd=%.0f  ty=%.3f°  tz=%.3f°",
                        bc_y, bc_z, lsd, ty_deg, tz_deg)
                    self.finished.emit(bc_y, bc_z, lsd, ty_deg, tz_deg)
                    return
                else:
                    # No rings → flat-detector sharpness metric (step 1) only.
                    # Unreliable for |tilt| > ~2° or spotty patterns.
                    # Load a calibrant for better results.
                    bc_y, bc_z = find_beam_center_sharpness(
                        self.image, mask, px=px,
                        bc_y_init=self.geom.get('bc_y'),
                        bc_z_init=self.geom.get('bc_z'))
                    log.info("Sharpness finder → bc_y=%.1f  bc_z=%.1f",
                             bc_y, bc_z)
            else:
                bc_y, bc_z, lsd = find_beam_center(
                    self.image, mask,
                    self.rings,
                    lsd=self.geom['lsd'], px=self.geom['px'],
                    bc_y_init=self.geom['bc_y'],
                    bc_z_init=self.geom['bc_z'])
            self.finished.emit(bc_y, bc_z, lsd, ty_deg, tz_deg)
        except Exception as exc:
            log.error("Find center failed:\n%s", traceback.format_exc())
            self.error.emit(str(exc))


class _CalibWorker(QThread):
    """Background worker that runs the multi-stage calibration optimizer.

    Runs calibrate() in a separate thread so the GUI stays responsive.
    Emits stage_done after each stage and finished when all stages complete.
    Raises _CalibrationCancelled internally when the user clicks Stop.

    Signals
    -------
    finished(dict)           Final result dict from calibrate().
    stage_done(int, dict)    (stage_index, intermediate result) after each stage.
    error(str)               Human-readable error message on failure.
    status_msg(str)          Per-iteration progress text for the status bar.
    progress(int, int)       (completed_iterations, total_iterations).
    """
    finished   = Signal(dict)
    stage_done = Signal(int, dict)
    error      = Signal(str)
    status_msg = Signal(str)
    progress   = Signal(int, int)

    def __init__(self, image, mask, geom, panel_map, rings_all,
                 tth_rough_max, optimize_panels, fit_distortion=False,
                 parent=None):
        super().__init__(parent)
        self.image           = image
        self.mask            = mask
        self.geom            = geom
        self.panel_map       = panel_map
        self.rings_all       = rings_all
        self.tth_rough_max   = tth_rough_max
        self.optimize_panels = optimize_panels
        self.fit_distortion  = fit_distortion

    def run(self):
        import traceback
        n_stages = 3 if self.optimize_panels else 2
        iter_schedule = [3, 5, 8] if self.optimize_panels else [3, 5]
        total_iters   = sum(iter_schedule)
        mask = (self.mask if self.mask is not None
                else np.zeros(self.image.shape, dtype=bool))
        try:
            rings_rough = ([r for r in self.rings_all
                            if r['tth'] <= self.tth_rough_max]
                           or self.rings_all[:4])

            def _cb(stage):
                offset = sum(iter_schedule[:stage - 1])
                def _inner(iteration, n_iter, strain_ppm, n_pts):
                    # Cooperative cancellation: raise so calibrate() exits early.
                    if self.isInterruptionRequested():
                        raise _CalibrationCancelled()
                    msg = (f"Stage {stage}/{n_stages}, iter {iteration}/{n_iter} — "
                           f"strain = {strain_ppm:.0f} ppm  ({n_pts} pts)")
                    self.status_msg.emit(msg)
                    self.progress.emit(offset + iteration, total_iters)
                return _inner

            self.progress.emit(0, total_iters)

            log.info("Calibration stage 1/%d: rough alignment (%d rings)",
                     n_stages, len(rings_rough))
            self.status_msg.emit(
                f"Stage 1/{n_stages}: rough alignment ({len(rings_rough)} rings)…")
            r1 = calibrate(
                self.image, mask, self.geom, self.panel_map,
                rings_rough, optimize_shifts=False,
                tth_tol_factor=10.0,
                tol_lsd=self.geom['lsd'] * 0.15,   # ±15 % — accommodates rough Lsd
                tol_bc=30.0, tol_tilts=5.0,          # wide enough for uncalibrated start
                tol_p3=0.0,   # never fit distortion in rough stage
                n_iterations=3, verbose=False,
                progress_cb=_cb(1))
            log.info("Stage 1 done — strain=%.0f ppm",
                     r1.get('mean_strain', float('nan')) * 1e6)
            self.stage_done.emit(1, r1)

            if self.isInterruptionRequested():
                raise _CalibrationCancelled()

            # Stage 2 strategy depends on which corrections are requested.
            # Key rule: panel shifts correct pixel coordinates *before* the
            # geometry transform; distortion is a smooth correction defined
            # on top of those corrected positions.  Fitting distortion before
            # panels causes the polynomial to absorb discrete per-module
            # offsets, contaminating both results.  Therefore:
            #   only panels   → Stage 2 = PONI only, Stage 3 = PONI + panels
            #   only distortion → Stage 2 = PONI + distortion  (no Stage 3)
            #   both           → Stage 2 = PONI + panels,
            #                    Stage 3 = PONI + panels + distortion
            both      = self.optimize_panels and self.fit_distortion
            s2_shifts = both   # panels in Stage 2 only when both are requested
            tol_p2_s2 = 0.002 if (self.fit_distortion and not both) else 0.0
            tol_p3_s2 = 45.0  if (self.fit_distortion and not both) else 0.0
            s2_desc   = ("panel shifts" if both
                         else f"full calibration ({len(self.rings_all)} rings)")
            log.info("Calibration stage 2/%d: %s", n_stages, s2_desc)
            self.status_msg.emit(f"Stage 2/{n_stages}: {s2_desc}…")
            r2 = calibrate(
                self.image, mask, r1['geom'], self.panel_map,
                self.rings_all, optimize_shifts=s2_shifts,
                tol_lsd=1000.0, tol_bc=5.0, tol_tilts=1.0,
                tol_p2=tol_p2_s2, tol_p3=tol_p3_s2,
                n_iterations=5, verbose=False,
                progress_cb=_cb(2))
            r2['distortion_fitted'] = self.fit_distortion and not both
            log.info("Stage 2 done — strain=%.0f ppm",
                     r2.get('mean_strain', float('nan')) * 1e6)
            self.stage_done.emit(2, r2)

            if not self.optimize_panels:
                self.finished.emit(r2)
                return

            if self.isInterruptionRequested():
                raise _CalibrationCancelled()

            log.info("Calibration stage 3/%d: panel shifts", n_stages)
            self.status_msg.emit(
                f"Stage 3/{n_stages}: panel shift optimization…")
            tol_p2_s3 = 0.001 if self.fit_distortion else 0.0
            tol_p3_s3 = 45.0  if self.fit_distortion else 0.0
            r3 = calibrate(
                self.image, mask, r2['geom'], self.panel_map,
                self.rings_all, optimize_shifts=True,
                tol_lsd=500.0, tol_bc=2.0, tol_tilts=0.5,
                tol_p2=tol_p2_s3, tol_p3=tol_p3_s3,
                n_iterations=8, verbose=False,
                progress_cb=_cb(3))
            r3['distortion_fitted'] = self.fit_distortion
            log.info("Stage 3 done — strain=%.0f ppm",
                     r3.get('mean_strain', float('nan')) * 1e6)
            self.stage_done.emit(3, r3)
            self.finished.emit(r3)
        except _CalibrationCancelled:
            log.info("Calibration stopped by user")
            # UI already updated by _on_stop — emit nothing
        except Exception as exc:
            log.error("Calibration failed:\n%s", traceback.format_exc())
            self.error.emit(str(exc))



# ── Ring selection dialog (shown before manual point picking) ─────────────────

class _RingPickDialog(QDialog):
    """Modal dialog: user picks which calibrant ring they will click on.

    Shows the full ring table from the loaded calibrant (2θ, d, hkl, I_rel).
    The user must select exactly one row.  This makes the ring assignment
    unambiguous — the program no longer has to guess.
    """

    def __init__(self, rings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Calibrant Ring")
        self.setModal(True)
        self._rings = rings
        self._selected_ring = None

        vl = QVBoxLayout(self)
        vl.setSpacing(8)
        vl.setContentsMargins(12, 12, 12, 10)

        instr = QLabel(
            "Select the ring whose arc you will click on.\n"
            "You need to know which reflection is visible on the detector.\n"
            "Double-click a row, or select and press OK.")
        instr.setWordWrap(True)
        vl.addWidget(instr)

        self._table = QTableWidget(len(rings), 6)
        self._table.setHorizontalHeaderLabels(
            ["2θ (deg)", "d (Å)", "h", "k", "l", "I (rel)"])
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)

        for i, ring in enumerate(rings):
            items = [
                QTableWidgetItem(f"{ring['tth']:.4f}"),
                QTableWidgetItem(f"{ring['d']:.4f}"),
                QTableWidgetItem(str(ring['h'])),
                QTableWidgetItem(str(ring['k'])),
                QTableWidgetItem(str(ring['l'])),
                QTableWidgetItem(f"{ring.get('intensity', 0):.0f}"),
            ]
            for j, it in enumerate(items):
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._table.setItem(i, j, it)

        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hdr.setStretchLastSection(True)
        self._table.setMinimumWidth(420)
        self._table.setMinimumHeight(300)
        vl.addWidget(self._table)

        hl = QHBoxLayout()
        hl.setSpacing(6)
        self._ok_btn  = QPushButton("OK — pick this ring")
        cancel_btn    = QPushButton("Cancel")
        self._ok_btn.setEnabled(False)
        self._ok_btn.clicked.connect(self._on_ok)
        cancel_btn.clicked.connect(self.reject)
        hl.addWidget(self._ok_btn)
        hl.addWidget(cancel_btn)
        vl.addLayout(hl)

        self._table.selectionModel().selectionChanged.connect(self._on_sel)
        self._table.doubleClicked.connect(self._on_ok)
        self.adjustSize()

    def _on_sel(self, *_):
        rows = self._table.selectionModel().selectedRows()
        self._ok_btn.setEnabled(len(rows) > 0)

    def _on_ok(self, *_):
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        self._selected_ring = self._rings[rows[0].row()]
        self.accept()

    def selected_ring(self):
        return self._selected_ring


# ── Manual Find Center dialog (modeless, shown during point picking) ──────────

class _ManualCenterDialog(QDialog):
    """Modeless floating dialog shown during manual beam-center picking.

    Stays above the main window so the user can still click on the image
    canvas behind it.
    """

    def __init__(self, ring=None, parent=None):
        """
        Parameters
        ----------
        ring : dict or None
            The calibrant ring the user will click on, as returned by
            load_calibrant().  Keys: 'tth', 'd', 'h', 'k', 'l'.
            When provided, the ring identity is shown prominently at the top.
        """
        super().__init__(parent,
                         Qt.WindowType.Dialog |
                         Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle("Manual Find Center — picking points")
        self.setModal(False)

        vl = QVBoxLayout(self)
        vl.setSpacing(10)
        vl.setContentsMargins(12, 12, 12, 10)

        if ring is not None:
            hkl = f"({ring['h']} {ring['k']} {ring['l']})"
            ring_lbl = QLabel(
                f"Ring:  {hkl}   "
                f"2θ = {ring['tth']:.4f} deg   "
                f"d = {ring['d']:.4f} Å")
            ring_lbl.setStyleSheet("font-weight: bold; color: #00e676;")
            vl.addWidget(ring_lbl)

        instr = QLabel(
            "Click on the image to select points along this ring arc.\n"
            "At least 3 points are required — more gives a better fit.\n"
            "Left-drag to zoom in; right-click to zoom out.")
        instr.setWordWrap(True)
        vl.addWidget(instr)

        self._count_lbl = QLabel("Points selected: 0")
        vl.addWidget(self._count_lbl)

        hl = QHBoxLayout()
        hl.setSpacing(6)
        self.done_btn   = QPushButton("Done (0 pts)")
        self.cancel_btn = QPushButton("Cancel")
        self.done_btn.setEnabled(False)
        hl.addWidget(self.done_btn)
        hl.addWidget(self.cancel_btn)
        vl.addLayout(hl)

        self.adjustSize()

    def update_count(self, n):
        self._count_lbl.setText(f"Points selected: {n}")
        self.done_btn.setText(f"Done ({n} pts)")
        self.done_btn.setEnabled(n >= 3)


# ── Zoom-capable full-canvas image widget ─────────────────────────────────────

class _ZoomCanvas(QWidget):
    """Full-canvas matplotlib image with scroll/ROI zoom, mask overlay,
    and a HistClimWidget side panel."""

    def __init__(self, aspect='equal', parent=None):
        super().__init__(parent)
        self._aspect     = aspect
        self._img_handle = None
        self._msk_handle = None    # RGBA mask overlay
        self._ring_arcs  = []
        self._zoom_stack = []
        self._roi_patch    = None
        self._roi_start    = None
        self._roi_start_px = None   # canvas-pixel start (for size threshold)
        self._dragging     = False
        self._image        = None    # raw numpy array
        self._home_xlim    = None    # limits at full zoom-out
        self._home_ylim    = None
        self._cmap         = 'gray_r'
        self._cake_msk_handle    = None  # red overlay for masked cake bins
        self._lut_ring_handle    = None  # pixel-accurate raw strain overlay (RGBA)
        self._strain_cb_ax       = None  # colorbar inset for raw strain overlay
        self._cake_strain_handle = None  # pixel-accurate cake strain overlay (RGBA)
        self._cake_strain_cb_ax  = None  # colorbar inset for cake strain overlay
        self._click_interceptor  = None  # callable(col, row) — point-picking mode

        self._build_ui()

    def _build_ui(self):
        vl = QVBoxLayout(self)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)

        # Top row: image canvas + histogram side panel
        top = QWidget()
        hl = QHBoxLayout(top)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(0)

        self.fig = Figure(facecolor=_BG_FIG)
        self.ax  = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_facecolor(_BG_AX)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for sp in self.ax.spines.values():
            sp.set_visible(False)

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.mpl_connect('scroll_event',         self._on_scroll)
        self.canvas.mpl_connect('button_press_event',   self._on_press)
        self.canvas.mpl_connect('motion_notify_event',  self._on_motion)
        self.canvas.mpl_connect('button_release_event', self._on_release)
        hl.addWidget(self.canvas, stretch=1)

        # Histogram contrast panel
        self.hist = HistClimWidget(width=110, parent=self)
        hl.addWidget(self.hist, stretch=0)

        vl.addWidget(top, stretch=1)

        # Toolbar spanning the full width (canvas + histogram)
        self._toolbar = _NavBar(self.canvas, self, coordinates=False)
        self._toolbar.setMaximumHeight(28)
        vl.addWidget(self._toolbar, stretch=0)

    # ── Display ───────────────────────────────────────────────────────────────

    def show_image(self, image, extent=None, xlabel='', ylabel=''):
        """Display *image*.  extent = [x0, x1, y0, y1] or None."""
        self._image   = image
        finite = image[np.isfinite(image)]
        finite = finite[finite != 0] if len(finite) > 0 else finite
        p2  = float(np.percentile(finite, 2))  if len(finite) > 0 else 0
        p98 = float(np.percentile(finite, 98)) if len(finite) > 0 else 1

        orig = 'lower' if extent is not None else 'upper'
        asp  = self._aspect

        self.ax.clear()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for sp in self.ax.spines.values():
            sp.set_visible(False)

        self._img_handle = self.ax.imshow(
            image, cmap=self._cmap, aspect=asp,
            origin=orig, extent=extent,
            vmin=p2, vmax=p98,
            interpolation='nearest')

        if xlabel:
            self.ax.set_xlabel(xlabel, color=_FG, fontsize=7)
            self.ax.xaxis.set_tick_params(labelbottom=True)
            self.ax.tick_params(colors=_FG, labelsize=6)
        if ylabel:
            self.ax.set_ylabel(ylabel, color=_FG, fontsize=7)
            self.ax.yaxis.set_tick_params(labelleft=True)

        # Compute home limits (full image, used for right-click reset and zoom-out stop)
        if extent is not None:
            self._home_xlim = (min(extent[0], extent[1]), max(extent[0], extent[1]))
            # preserve orientation for y (imshow origin='lower' means y0 < y1)
            self._home_ylim = (extent[2], extent[3])
        else:
            h, w = image.shape[:2]
            self._home_xlim = (-0.5, w - 0.5)
            self._home_ylim = (h - 0.5, -0.5)   # origin='upper'

        # Lock axes limits so overlays and ring lines cannot expand them
        self.ax.set_xlim(self._home_xlim)
        self.ax.set_ylim(self._home_ylim)
        self.ax.autoscale(False)

        self._msk_handle = None
        self._ring_arcs.clear()
        self._zoom_stack.clear()
        self.canvas.draw_idle()

        self.hist.set_data(
            image, vmin=p2, vmax=p98,
            callback=self._on_clim_changed)

    def show_mask_overlay(self, mask, visible=True):
        """Draw semi-transparent red over masked (=1) pixels."""
        if self._image is None:
            return
        if self._msk_handle is not None:
            try:
                self._msk_handle.remove()
            except Exception:
                pass
            self._msk_handle = None

        if mask is None or not visible:
            self.canvas.draw_idle()
            return

        rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
        rgba[..., 0] = 1.0          # red
        rgba[..., 3] = 0.65 * (mask > 0)
        self._msk_handle = self.ax.imshow(
            rgba, origin='upper',
            interpolation='nearest', zorder=3)
        self.canvas.draw_idle()

    def show_cake_mask_overlay(self, rgba, extent, visible=True):
        """Display a precomputed RGBA mask overlay on the cake.

        rgba    : (n_eta, n_tth, 4) float32 array, or None.
        extent  : [tth_l, tth_r, eta_b, eta_t] outer-bin-edge extent.
        visible : if False, remove any existing overlay.
        """
        if hasattr(self, '_cake_msk_handle') and self._cake_msk_handle is not None:
            try:
                self._cake_msk_handle.remove()
            except Exception:
                pass
            self._cake_msk_handle = None

        if not visible or rgba is None:
            self.canvas.draw_idle()
            return

        self._cake_msk_handle = self.ax.imshow(
            rgba, origin='lower', aspect='auto',
            extent=extent, interpolation='nearest', zorder=4)
        self.canvas.draw_idle()

    def draw_rings(self, ring_list, bc_y, bc_z, lsd, px, nrows, ncols,
                   ty_deg=0.0, tz_deg=0.0):
        """Overlay calibrant ring positions using the full tilt-aware geometry.

        For a tilted detector rings appear as ellipses; this method ray-traces
        each ring from the sample through the tilt-rotated detector plane so
        the overlay matches the measured pattern for any detector orientation.
        """
        from ..geometry import build_tilt_matrix

        for arc in self._ring_arcs:
            try:
                arc.remove()
            except Exception:
                pass
        self._ring_arcs.clear()

        TRs   = build_tilt_matrix(0.0, ty_deg, tz_deg)
        n_det = TRs[:, 0]          # detector normal in lab frame

        n_pts = 720
        eta_v = np.linspace(-np.pi, np.pi, n_pts)

        # Diffracted beam unit vectors (fixed angular part, vary per ring below)
        # d_lab = (cos(tth), -sin(tth)*cos(eta), sin(tth)*sin(eta))
        # eta=0 → 3 o'clock (right), eta=90° → 12 o'clock (top)
        cos_eta = np.cos(eta_v)
        sin_eta = np.sin(eta_v)

        for ring in ring_list:
            tth = np.radians(ring['tth'])
            cos_tth = np.cos(tth)
            sin_tth = np.sin(tth)

            dx = np.full(n_pts, cos_tth)
            dy = -sin_tth * cos_eta
            dz =  sin_tth * sin_eta
            d = np.vstack([dx, dy, dz])     # 3 × n_pts

            # Intersection with tilted detector plane: t = lsd / (n_det · d)
            n_dot_d = n_det @ d             # (n_pts,)
            forward = n_dot_d > 1e-12       # exclude backward-scattered rays
            t = np.where(forward, lsd / n_dot_d, np.nan)

            # Lab hit position → detector frame via TRs^T
            r_hit      = t[np.newaxis, :] * d   # 3 × n_pts
            det_coords = TRs.T @ r_hit           # 3 × n_pts
            Yc = det_coords[1]                   # µm from BC (positive = left)
            Zc = det_coords[2]                   # µm from BC (positive = up)

            # Pixel coordinates: Yc positive → col to the left; Zc positive → row up
            xs = bc_y - Yc / px
            ys = bc_z - Zc / px

            inside = (forward & (xs >= 0) & (xs < ncols)
                              & (ys >= 0) & (ys < nrows))
            xs[~inside] = np.nan
            ys[~inside] = np.nan
            if inside.any():
                ln, = self.ax.plot(xs, ys,
                                   color=_XH2, lw=0.7, alpha=0.75, ls='-')
                self._ring_arcs.append(ln)
        self.canvas.draw_idle()

    def show_lut_ring_overlay(self, tth_lut, ring_list, tol_deg,
                              mask=None, visible=True):
        """Pixel-accurate ring overlay colored by signed local 2θ strain (ppm).

        Blue = measured 2θ below ideal (d expanded), red = above (d compressed).
        Returns (vmin_ppm, vmax_ppm) of the symmetric colormap range, or (nan, nan).

        Parameters
        ----------
        mask : 2-D array, 1 = bad pixel, 0 = good (optional).
               Masked pixels are excluded from the overlay so gap/dead pixels
               do not appear in the strain map.
        """
        import matplotlib.cm as _cm
        import matplotlib.colors as _mc
        from matplotlib.cm import ScalarMappable

        # Remove old overlay and colorbar
        for handle_attr in ('_lut_ring_handle', '_strain_cb_ax'):
            obj = getattr(self, handle_attr, None)
            if obj is not None:
                try:
                    obj.remove()
                except Exception:
                    pass
                setattr(self, handle_attr, None)

        if tth_lut is None or not ring_list:
            self.canvas.draw_idle()
            return float('nan'), float('nan')

        # Build signed strain map (ppm) — NaN outside ring bands
        strain_map = np.full(tth_lut.shape, np.nan, dtype=np.float32)
        for ring in ring_list:
            diff = tth_lut - ring['tth']
            in_ring = np.abs(diff) < tol_deg
            if in_ring.any():
                strain_map[in_ring] = (diff[in_ring] / ring['tth']) * 1e6

        # Blank out masked pixels so gap/dead regions are transparent.
        if mask is not None:
            strain_map[mask != 0] = np.nan

        valid = strain_map[np.isfinite(strain_map)]
        if len(valid) == 0:
            self.canvas.draw_idle()
            return float('nan'), float('nan')

        # Symmetric range: use 2nd/98th percentiles, floor at ±500 ppm
        vmax = max(abs(float(np.percentile(valid, 2))),
                   abs(float(np.percentile(valid, 98))),
                   500.0)

        cmap = _mc.LinearSegmentedColormap.from_list(
            'strain_bgr',
            ['#2166ac', _XH, '#d6604d'],  # blue → green → red
            N=256)
        norm = _mc.Normalize(vmin=-vmax, vmax=vmax)
        rgba = cmap(norm(strain_map)).astype(np.float32)
        rgba[..., 3] = np.where(np.isfinite(strain_map), 0.75, 0.0)

        self._lut_ring_handle = self.ax.imshow(
            rgba, origin='upper',
            interpolation='nearest', zorder=5)
        self._lut_ring_handle.set_visible(visible)

        # Colorbar inset at top-right of image axes
        try:
            self._strain_cb_ax = self.ax.inset_axes([0.74, 0.935, 0.24, 0.05])
            cb = self._strain_cb_ax.figure.colorbar(
                ScalarMappable(norm=norm, cmap=cmap),
                cax=self._strain_cb_ax, orientation='horizontal')
            cb.set_ticks([-vmax, 0, vmax])
            cb.set_ticklabels([f'{-vmax:.0f}', '0', f'{vmax:.0f}'])
            self._strain_cb_ax.tick_params(labelsize=9, colors=_FG, length=3)
            self._strain_cb_ax.set_title(
                'strain (ppm)', fontsize=8, color=_FG, pad=2)
            self._strain_cb_ax.set_visible(visible)
        except Exception:
            self._strain_cb_ax = None

        self.canvas.draw_idle()
        return -vmax, vmax

    def show_cake_strain_overlay(self, tth_c, ring_list, tol_deg, extent,
                                  px_cnt_cake=None, visible=True):
        """Strain-band overlay on the cake (2θ vs η) image.

        Each cell at 2θ within tol_deg of a calibrant ring is coloured by
        (2θ_col − 2θ_ring) / 2θ_ring in ppm.  When px_cnt_cake (n_eta × n_tth)
        is supplied, cells with zero valid pixels are transparent — matching the
        per-pixel masking applied by the raw-image strain overlay.  Without it,
        a uniform colour band is drawn across all η rows.
        """
        import matplotlib.colors as _mc
        from matplotlib.cm import ScalarMappable

        for attr in ('_cake_strain_handle', '_cake_strain_cb_ax'):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.remove()
                except Exception:
                    pass
                setattr(self, attr, None)

        if tth_c is None or not ring_list:
            self.canvas.draw_idle()
            return

        strain_row = np.full(len(tth_c), np.nan, dtype=np.float32)
        for ring in ring_list:
            diff = tth_c - ring['tth']
            in_band = np.abs(diff) < tol_deg
            strain_row[in_band] = (diff[in_band] / ring['tth']) * 1e6

        valid = strain_row[np.isfinite(strain_row)]
        if len(valid) == 0:
            self.canvas.draw_idle()
            return

        vmax = max(abs(float(np.percentile(valid, 2))),
                   abs(float(np.percentile(valid, 98))),
                   500.0)

        cmap = _mc.LinearSegmentedColormap.from_list(
            'strain_bgr', ['#2166ac', _XH, '#d6604d'], N=256)
        norm = _mc.Normalize(vmin=-vmax, vmax=vmax)

        rgba_1d = cmap(norm(strain_row)).astype(np.float32)
        rgba_1d[np.isnan(strain_row), 3] = 0.0
        if px_cnt_cake is not None:
            n_eta = px_cnt_cake.shape[0]
            rgba_2d = np.tile(rgba_1d[np.newaxis, :, :], (n_eta, 1, 1))
            rgba_2d[px_cnt_cake == 0, 3] = 0.0
        else:
            rgba_2d = rgba_1d[np.newaxis, :, :]
        rgba_2d[..., 3] *= 0.75

        self._cake_strain_handle = self.ax.imshow(
            rgba_2d, origin='lower', extent=extent,
            aspect='auto', interpolation='nearest', zorder=5)
        self._cake_strain_handle.set_visible(visible)

        try:
            self._cake_strain_cb_ax = self.ax.inset_axes([0.74, 0.935, 0.24, 0.05])
            cb = self._cake_strain_cb_ax.figure.colorbar(
                ScalarMappable(norm=norm, cmap=cmap),
                cax=self._cake_strain_cb_ax, orientation='horizontal')
            cb.set_ticks([-vmax, 0, vmax])
            cb.set_ticklabels([f'{-vmax:.0f}', '0', f'{vmax:.0f}'])
            self._cake_strain_cb_ax.tick_params(labelsize=9, colors=_FG, length=3)
            self._cake_strain_cb_ax.set_title(
                'strain (ppm)', fontsize=8, color=_FG, pad=2)
            self._cake_strain_cb_ax.set_visible(visible)
        except Exception:
            self._cake_strain_cb_ax = None

        self.canvas.draw_idle()

    def set_strain_visible(self, visible):
        """Show or hide all strain overlay handles on this canvas."""
        changed = False
        for attr in ('_lut_ring_handle', '_strain_cb_ax',
                     '_cake_strain_handle', '_cake_strain_cb_ax'):
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.set_visible(visible)
                changed = True
        if changed:
            self.canvas.draw_idle()

    def clear_rings(self):
        for arc in self._ring_arcs:
            try:
                arc.remove()
            except Exception:
                pass
        self._ring_arcs.clear()
        for handle_attr in ('_lut_ring_handle', '_strain_cb_ax',
                            '_cake_strain_handle', '_cake_strain_cb_ax'):
            obj = getattr(self, handle_attr, None)
            if obj is not None:
                try:
                    obj.remove()
                except Exception:
                    pass
                setattr(self, handle_attr, None)
        self.canvas.draw_idle()

    def clear(self):
        self.ax.clear()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for sp in self.ax.spines.values():
            sp.set_visible(False)
        self._img_handle         = None
        self._msk_handle         = None
        self._cake_msk_handle    = None
        self._lut_ring_handle    = None
        self._strain_cb_ax       = None
        self._cake_strain_handle = None
        self._cake_strain_cb_ax  = None
        self._ring_arcs.clear()
        self._zoom_stack.clear()
        self.hist.clear()
        self.canvas.draw_idle()

    def _on_clim_changed(self, vmin, vmax):
        if self._img_handle is not None and vmin < vmax:
            self._img_handle.set_clim(vmin, vmax)
            self.canvas.draw_idle()

    def apply_mpl_theme(self, colors):
        self._cmap = colors.get('cmap', 'gray_r')
        fg    = colors.get('fg',    _FG)
        bg_ax = colors.get('ax',    _BG_AX)
        spine = colors.get('spine', '#999999')
        self.fig.set_facecolor(colors.get('fig', _BG_FIG))
        _style_ax(self.ax, fg=fg, bg=bg_ax, spine=spine)
        if self._img_handle is not None:
            self._img_handle.set_cmap(self._cmap)
        self.canvas.draw_idle()
        self.hist.apply_mpl_theme(colors)
        _refresh_navbar_icons(self._toolbar)

    # ── Zoom ─────────────────────────────────────────────────────────────────

    def _reset_zoom(self):
        """Fit image to display panel — right-click or reached minimum zoom."""
        if self._home_xlim is not None:
            self.ax.set_xlim(self._home_xlim)
            self.ax.set_ylim(self._home_ylim)
        self.ax.set_aspect(self._aspect)   # restore home aspect ratio
        self._zoom_stack.clear()
        self.canvas.draw_idle()

    def _on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        factor = 1.35
        scale  = 1.0 / factor if event.button == 'up' else factor

        xc, yc = event.xdata, event.ydata
        xlim = list(self.ax.get_xlim())
        ylim = list(self.ax.get_ylim())
        new_x = [xc + (x - xc) * scale for x in xlim]
        new_y = [yc + (y - yc) * scale for y in ylim]

        if event.button == 'down' and self._home_xlim is not None:
            home_xspan = abs(self._home_xlim[1] - self._home_xlim[0])
            new_xspan  = abs(new_x[1] - new_x[0])
            if new_xspan >= home_xspan:
                self._reset_zoom()
                return

        self._zoom_stack.append((xlim, ylim))
        self.ax.set_xlim(new_x)
        self.ax.set_ylim(new_y)
        self.canvas.draw_idle()

    def _on_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 3:
            # Right-click always resets to full image fit
            self._reset_zoom()
            return
        if getattr(self.canvas, 'toolbar', None) and self.canvas.toolbar.mode:
            return
        if event.button == 1:
            self._dragging     = True
            self._roi_start    = (event.xdata, event.ydata)
            self._roi_start_px = (event.x, event.y)
            if self._roi_patch is not None:
                self._roi_patch.remove()
                self._roi_patch = None

    def _on_motion(self, event):
        if not self._dragging or self._roi_start is None:
            return
        if event.inaxes != self.ax or event.xdata is None:
            return
        x0, y0 = self._roi_start
        dx = event.xdata - x0
        dy = event.ydata - y0
        if self._roi_patch is not None:
            self._roi_patch.remove()
        self._roi_patch = Rectangle(
            (x0, y0), dx, dy,
            lw=1, edgecolor=_XH, facecolor='none', ls='--', alpha=0.85)
        self.ax.add_patch(self._roi_patch)
        self.canvas.draw_idle()

    def _on_release(self, event):
        if not self._dragging or self._roi_start is None:
            return
        self._dragging = False
        if event.inaxes != self.ax or event.xdata is None:
            self._clear_roi()
            return

        x0, y0 = self._roi_start
        dx = event.xdata - x0
        dy = event.ydata - y0
        # Use canvas-pixel distance so the threshold works for any data units
        if self._roi_start_px is not None:
            px_ok = (abs(event.x - self._roi_start_px[0]) >= 5 and
                     abs(event.y - self._roi_start_px[1]) >= 5)
        else:
            px_ok = (abs(dx) > 0 and abs(dy) > 0)
        if not px_ok:
            self._clear_roi()
            # Gesture was a click (not a drag) — register point if interceptor active
            if self._click_interceptor is not None and event.xdata is not None:
                self._click_interceptor(event.xdata, event.ydata)
            return

        self._zoom_stack.append((self.ax.get_xlim(), self.ax.get_ylim(),
                                  self.ax.get_aspect()))
        x0n, x1n = sorted([x0, x0 + dx])
        y0n, y1n = sorted([y0, y0 + dy])
        # ROI zoom fills the panel — switch to auto aspect
        self.ax.set_aspect('auto')
        self.ax.set_xlim(x0n, x1n)
        # Preserve y-axis orientation (inverted for origin='upper', normal for 'lower')
        cur_ylim = self.ax.get_ylim()
        if cur_ylim[0] > cur_ylim[1]:   # inverted (raw image)
            self.ax.set_ylim(y1n, y0n)
        else:                            # normal (cake)
            self.ax.set_ylim(y0n, y1n)
        self._clear_roi()

    def _clear_roi(self):
        if self._roi_patch is not None:
            self._roi_patch.remove()
            self._roi_patch = None
        self._roi_start    = None
        self._roi_start_px = None
        self.canvas.draw_idle()


# ── CalibrationTab ────────────────────────────────────────────────────────────

class CalibrationTab(QWidget):
    """Calibration tab: QFormLayout left panel + toggled image right panel."""

    geometry_ready = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._image         = None   # loaded TIFF
        self._image_path    = None   # path of the loaded TIFF (for save defaults)
        self._mask          = None   # 0=good 1=bad
        self._rings         = []     # calibrant ring dicts
        self._geom          = None   # best geometry dict
        self._calib_result  = None   # full calibration result dict
        self._tth_lut       = None
        self._eta_lut       = None
        self._cake_tth      = None   # 2θ axis from last _ensure_cake()
        self._cake_eta      = None   # η  axis from last _ensure_cake()
        self._mask_visible   = False
        self._strain_visible = True  # whether strain overlay is shown
        self._calib_worker  = None
        self._fc_worker     = None   # find-center background worker
        self._lineout    = None   # (tth, I, bg, I_sub, sigma, px_cnt)
        self._lo_show       = {'I': True, 'I_sub': False, 'SNIP': False}
        self._lo_show_sigma = True
        # Current matplotlib theme colours — initialised to system defaults,
        # updated by apply_mpl_theme() on every theme switch.
        self._mpl_colors = {
            'fig': _BG_FIG, 'ax': _BG_AX, 'fg': _FG, 'spine': '#999999',
            'line_I': '#1565c0', 'line_Isub': '#00695c',
            'line_snip': '#bf360c', 'ring': '#b71c1c',
        }
        self._cake_ring_lines = []   # axvline handles on cake canvas
        self._panel_map    = None   # stored when calibration starts
        self._panel_shifts = None   # final panel shifts from calibrate()

        # Manual Find Center state
        self._manual_center_mode = False
        self._manual_center_ring = None  # ring dict chosen in _RingPickDialog
        self._center_points      = []   # list of (col, row) pixel coords
        self._center_markers     = []   # matplotlib artist handles
        self._mc_dialog          = None  # modeless picking dialog

        # Lineout zoom state
        self._lo_zoom_stack  = []
        self._lo_home_xlim   = None
        self._lo_home_ylim   = None
        self._lo_dragging    = False
        self._lo_roi_start   = None
        self._lo_roi_start_px = None
        self._lo_roi_patch   = None

        self._build_ui()

    # ── Top-level layout ──────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_left_panel(),  stretch=0)
        root.addWidget(self._build_right_panel(), stretch=1)

    # ── Left panel ────────────────────────────────────────────────────────────

    def _build_left_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(300)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        vl = QVBoxLayout(inner)
        vl.setContentsMargins(6, 6, 6, 6)
        vl.setSpacing(6)

        vl.addWidget(self._build_detector_group())
        vl.addWidget(self._build_beam_group())
        vl.addWidget(self._build_geometry_group())
        vl.addWidget(self._build_calibrant_group())
        vl.addWidget(self._build_actions_group())
        vl.addWidget(self._build_results_group())
        vl.addStretch()

        scroll.setWidget(inner)
        return scroll

    def _form_group(self, title):
        """QGroupBox + QFormLayout pair."""
        grp  = QGroupBox(title)
        form = QFormLayout(grp)
        form.setSpacing(4)
        form.setContentsMargins(6, 8, 6, 6)
        return grp, form

    def _build_detector_group(self):
        grp, form = self._form_group("Detector")

        self._det_combo = QComboBox()
        self._det_combo.addItems(DETECTOR_NAMES)
        idx = DETECTOR_NAMES.index(DEFAULT_DETECTOR) if DEFAULT_DETECTOR in DETECTOR_NAMES else 0
        self._det_combo.setCurrentIndex(idx)
        self._det_combo.currentIndexChanged.connect(self._on_detector_changed)
        form.addRow("Model:", self._det_combo)

        det = DETECTORS[DETECTOR_NAMES[idx]]
        self._nrows_ed = QLineEdit(str(det['nrows']))
        self._ncols_ed = QLineEdit(str(det['ncols']))
        self._px_ed    = QLineEdit(str(det['px_um']))
        form.addRow("Rows:", self._nrows_ed)
        form.addRow("Cols:", self._ncols_ed)
        form.addRow("px (µm):", self._px_ed)
        return grp

    def _build_beam_group(self):
        grp, form = self._form_group("Beam")

        self._energy_ed = QLineEdit("")
        self._energy_ed.setPlaceholderText("e.g. 25000")
        self._energy_ed.editingFinished.connect(self._sync_energy_to_lambda)
        self._energy_ed.editingFinished.connect(self._auto_refresh_rings)
        form.addRow("E (eV):", self._energy_ed)

        self._lambda_ed = QLineEdit("")
        self._lambda_ed.setPlaceholderText("e.g. 0.49594")
        self._lambda_ed.editingFinished.connect(self._sync_lambda_to_energy)
        self._lambda_ed.editingFinished.connect(self._auto_refresh_rings)
        form.addRow("λ (Å):", self._lambda_ed)
        return grp

    def _build_geometry_group(self):
        container = QWidget()
        vl = QVBoxLayout(container)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(2)

        self._geom_toggle = QPushButton("▼  Geometry")
        self._geom_toggle.setCheckable(True)
        self._geom_toggle.setChecked(True)
        self._geom_toggle.setStyleSheet(
            "QPushButton { text-align: left; padding-left: 6px; font-weight: bold; }")
        self._geom_toggle.toggled.connect(self._on_geom_toggled)
        vl.addWidget(self._geom_toggle)

        self._geom_body = QGroupBox()
        self._geom_body.setFlat(True)
        form = QFormLayout(self._geom_body)
        form.setSpacing(4)
        form.setContentsMargins(6, 4, 6, 4)

        # Initialise BC to detector centre; tilts to zero — no beamline assumptions
        det = DETECTORS[DETECTOR_NAMES[self._det_combo.currentIndex()]]
        bcy_default = det['ncols'] / 2.0
        bcz_default = det['nrows'] / 2.0

        self._lsd_ed = QLineEdit("")
        self._lsd_ed.setPlaceholderText("e.g. 300000")
        self._bcy_ed = QLineEdit(f"{bcy_default:.1f}")
        self._bcz_ed = QLineEdit(f"{bcz_default:.1f}")
        self._tx_ed  = QLineEdit("0.0")
        self._ty_ed  = QLineEdit("0.0")
        self._tz_ed  = QLineEdit("0.0")

        form.addRow("Lsd (µm):", self._lsd_ed)
        form.addRow("BC_Y (px):", self._bcy_ed)
        form.addRow("BC_Z (px):", self._bcz_ed)
        form.addRow("tx (deg):", self._tx_ed)
        form.addRow("ty (deg):", self._ty_ed)
        form.addRow("tz (deg):", self._tz_ed)
        vl.addWidget(self._geom_body)
        return container

    def _build_calibrant_group(self):
        grp, form = self._form_group("Calibrant")

        self._cal_combo = QComboBox()
        cals = _list_calibrants()
        self._cal_combo.addItems(cals)
        # Default to CeO2 (case-insensitive match)
        ceo2_match = next((c for c in cals if c.lower() == 'ceo2'), None)
        if ceo2_match:
            self._cal_combo.setCurrentText(ceo2_match)
        self._cal_combo.currentIndexChanged.connect(self._auto_refresh_rings)
        form.addRow("Material:", self._cal_combo)

        self._tth_max_ed  = QLineEdit("20")
        self._tth_max_ed.editingFinished.connect(self._auto_refresh_rings)
        self._rough_max_ed = QLineEdit("8")
        form.addRow("2θ max (deg):", self._tth_max_ed)
        form.addRow("Rough ≤ (deg):", self._rough_max_ed)
        return grp

    def _build_actions_group(self):
        grp = QGroupBox("Actions")
        vl  = QVBoxLayout(grp)
        vl.setSpacing(4)
        vl.setContentsMargins(6, 8, 6, 6)

        def row(*btns):
            hl = QHBoxLayout()
            for b in btns:
                hl.addWidget(b)
            return hl

        self._btn_load_tiff   = QPushButton("Load Image…")
        self._btn_mask_editor = QPushButton("Mask…")
        self._btn_load_tiff.clicked.connect(self._on_load_tiff)
        self._btn_mask_editor.clicked.connect(self._on_open_mask_editor)
        vl.addLayout(row(self._btn_load_tiff, self._btn_mask_editor))

        self._btn_find_center_auto = QPushButton("Auto Find Center")
        self._btn_manual_center    = QPushButton("Manual Find Center")
        self._btn_find_center_auto.setToolTip(
            "Estimate beam centre, Lsd, and tilts automatically.\n\n"
            "With calibrant loaded (recommended):\n"
            "  Uses the full tilt-aware geometry model — rings that appear as\n"
            "  ellipses on a tilted detector are correctly handled.\n"
            "  Scans (ty, tz) then (bc_y, bc_z), then refines all 5 parameters\n"
            "  jointly via Nelder-Mead.  Works for large tilts and spotty patterns.\n\n"
            "Without calibrant:\n"
            "  Flat-detector sharpness metric only — unreliable for |tilt| > ~2\u00b0\n"
            "  or spotty calibrants.  Load a calibrant for better results.")
        self._btn_manual_center.setToolTip(
            "Use this for difficult cases:\n"
            "  \u2022 detector is heavily off-axis (beam near edge or outside)\n"
            "  \u2022 only one or two arcs are visible, possibly segmented\n"
            "  \u2022 the visible ring is not the first Bragg reflection\n"
            "  \u2022 Auto Find Center gave a poor result\n\n"
            "You select the hkl ring from the calibrant table, then click\n"
            "\u22653 points on that arc. The beam centre and Lsd are solved\n"
            "exactly from your ring assignment \u2014 no guessing.")
        self._btn_find_center_auto.clicked.connect(self._on_find_center_auto)
        self._btn_manual_center.clicked.connect(self._on_manual_center_start)
        vl.addLayout(row(self._btn_find_center_auto, self._btn_manual_center))

        self._chk_panels     = QCheckBox("Optimize panels")
        self._chk_panels.setToolTip(
            "Refine per-module rigid-body shifts (dY, dZ, dLsd, dTheta) for\n"
            "tiled detectors (Pilatus, Eiger). Panel shifts are saved in the\n"
            "geometry TOML and reloaded automatically via Load Geometry.")
        vl.addWidget(self._chk_panels)
        self._chk_distortion = QCheckBox("Fit distortion (p2, p3)")
        self._chk_distortion.setToolTip(
            "Fit p2 (isotropic radial distortion) and p3 (azimuthal phase of\n"
            "the 4-fold term). Leave unchecked unless per-ring strain residuals\n"
            "show a systematic radial or azimuthal trend after flat-geometry\n"
            "calibration. For Pilatus/Eiger sensors this is rarely needed.\n\n"
            "If both 'Optimize panels' and 'Fit distortion' are checked:\n"
            "  Stage 2 fits panel shifts first (they correct pixel positions).\n"
            "  Stage 3 then refines distortion on top of the corrected panels.\n"
            "  Fitting distortion before panels would cause the polynomial to\n"
            "  absorb per-module offsets, giving wrong values for both.")
        vl.addWidget(self._chk_distortion)

        self._btn_calibrate = QPushButton("Run Calibration")
        self._btn_stop      = QPushButton("Stop")
        self._btn_stop.setEnabled(False)
        self._btn_calibrate.clicked.connect(self._on_calibrate)
        self._btn_stop.clicked.connect(self._on_stop)
        vl.addLayout(row(self._btn_calibrate, self._btn_stop))

        self._btn_reset = QPushButton("Reset")
        self._btn_reset.setToolTip(
            "Clear image, mask, rings, and calibration — restart from scratch")
        self._btn_reset.clicked.connect(self._on_reset)
        vl.addWidget(self._btn_reset)

        self._btn_show_result = QPushButton("Show/Save Result…")
        self._btn_use_geom    = QPushButton("Send to Integration")
        self._btn_show_result.clicked.connect(self._on_show_result)
        self._btn_use_geom.clicked.connect(self._on_use_geometry)
        vl.addWidget(self._btn_show_result)
        vl.addWidget(self._btn_use_geom)
        return grp

    def _build_results_group(self):
        grp = QGroupBox("Results")
        vl  = QVBoxLayout(grp)
        vl.setContentsMargins(6, 8, 6, 6)
        self._results_lbl = QLabel("No calibration run yet.")
        self._results_lbl.setWordWrap(True)
        self._results_lbl.setStyleSheet(f"color: {_FG}; font-size: 10px;")
        vl.addWidget(self._results_lbl)
        return grp

    # ── Right panel ───────────────────────────────────────────────────────────

    def _build_right_panel(self):
        panel = QWidget()
        vl = QVBoxLayout(panel)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)
        vl.addWidget(self._build_display_header())
        vl.addWidget(self._build_stacked_display(), stretch=1)
        return panel

    def _build_display_header(self):
        bar = QWidget()
        bar.setFixedHeight(30)
        hl = QHBoxLayout(bar)
        hl.setContentsMargins(4, 2, 4, 2)
        hl.setSpacing(4)

        def _btn(text, w=56):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setFixedWidth(w)
            return b

        self._btn_raw     = _btn("Raw")
        self._btn_cake    = _btn("Cake")
        self._btn_lineout = _btn("Lineout", 64)
        self._btn_mask    = _btn("Mask ○", 60)
        self._btn_strain  = _btn("Strain ○", 64)

        self._btn_raw.setChecked(True)
        self._btn_strain.setChecked(True)
        self._btn_raw.clicked.connect(lambda: self._set_view('raw'))
        self._btn_cake.clicked.connect(lambda: self._set_view('cake'))
        self._btn_lineout.clicked.connect(lambda: self._set_view('lineout'))
        self._btn_mask.clicked.connect(self._on_toggle_mask)
        self._btn_strain.clicked.connect(self._on_toggle_strain)

        self._pos_label = QLabel("  —  ")
        self._pos_label.setStyleSheet(
            "color: #ffffff; font-size: 13px; padding-left: 4px;")

        hl.addWidget(self._btn_raw)
        hl.addWidget(self._btn_cake)
        hl.addWidget(self._btn_lineout)
        hl.addSpacing(8)
        hl.addWidget(self._btn_mask)
        hl.addWidget(self._btn_strain)
        hl.addSpacing(8)
        hl.addWidget(self._pos_label)
        hl.addStretch()
        return bar

    def _build_stacked_display(self):
        self._stack = QStackedWidget()

        # Page 0: Raw  — aspect='equal' preserves true pixel aspect ratio
        self._raw_canvas = _ZoomCanvas(aspect='equal')
        self._raw_canvas.canvas.mpl_connect(
            'motion_notify_event', self._on_raw_motion)
        self._stack.addWidget(self._raw_canvas)

        # Page 1: Cake
        self._cake_canvas = _ZoomCanvas(aspect='auto')
        self._cake_canvas.canvas.mpl_connect(
            'motion_notify_event', self._on_cake_motion)
        self._stack.addWidget(self._cake_canvas)

        # Page 2: Lineout
        self._lineout_widget = self._build_lineout_page()
        self._stack.addWidget(self._lineout_widget)

        return self._stack

    def _build_lineout_page(self):
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)

        vl.addWidget(self._build_lineout_header())

        self.line_fig = Figure(facecolor=_BG_FIG)
        self.line_ax  = self.line_fig.add_axes([0.10, 0.14, 0.87, 0.78])
        _style_ax(self.line_ax)
        self.line_ax.set_xlabel('2θ (deg)', fontsize=8)
        self.line_ax.set_ylabel('Intensity', fontsize=8)
        self.line_ax.set_title('Lineout', fontsize=9)
        self.line_canvas = FigureCanvasQTAgg(self.line_fig)
        self.line_canvas.mpl_connect('scroll_event',         self._on_lo_scroll)
        self.line_canvas.mpl_connect('button_press_event',   self._on_lo_press)
        self.line_canvas.mpl_connect('motion_notify_event',  self._on_lo_motion)
        self.line_canvas.mpl_connect('motion_notify_event',  self._on_lo_position)
        self.line_canvas.mpl_connect('button_release_event', self._on_lo_release)
        vl.addWidget(self.line_canvas, stretch=1)

        self._lo_toolbar = _NavBar(self.line_canvas, w, coordinates=False)
        self._lo_toolbar.setMaximumHeight(28)
        vl.addWidget(self._lo_toolbar)
        return w

    def _build_lineout_header(self):
        bar = QWidget()
        bar.setFixedHeight(30)
        hl = QHBoxLayout(bar)
        hl.setContentsMargins(4, 2, 4, 2)
        hl.setSpacing(4)

        def _btn(text, w=52):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setFixedWidth(w)
            return b

        self._btn_lo_I     = _btn("I",    36)
        self._btn_lo_Isub  = _btn("I_sub", 52)
        self._btn_lo_snip  = _btn("SNIP",  48)
        self._btn_lo_sigma = _btn("±σ",   34)
        self._btn_lo_I.setChecked(True)
        self._btn_lo_sigma.setChecked(True)
        self._btn_lo_sigma.setToolTip("Show / hide ±1σ Poisson error band")
        self._btn_lo_I.clicked.connect(    lambda checked: self._toggle_lo_curve('I',     checked))
        self._btn_lo_Isub.clicked.connect( lambda checked: self._toggle_lo_curve('I_sub', checked))
        self._btn_lo_snip.clicked.connect( lambda checked: self._toggle_lo_curve('SNIP',  checked))
        self._btn_lo_sigma.clicked.connect(lambda checked: self._toggle_lo_sigma(checked))

        self._lo_pos_label = QLabel("  —  ")
        self._lo_pos_label.setStyleSheet(
            "color: #ffffff; font-size: 13px; padding-left: 8px;")

        hl.addWidget(self._btn_lo_I)
        hl.addWidget(self._btn_lo_Isub)
        hl.addWidget(self._btn_lo_snip)
        hl.addWidget(self._btn_lo_sigma)
        hl.addSpacing(8)
        hl.addWidget(self._lo_pos_label)
        hl.addStretch()
        return bar

    # ── Toggle helpers ────────────────────────────────────────────────────────

    def _set_view(self, view):
        self._btn_raw.setChecked(view == 'raw')
        self._btn_cake.setChecked(view == 'cake')
        self._btn_lineout.setChecked(view == 'lineout')
        if view == 'raw':
            self._stack.setCurrentIndex(0)
        elif view == 'cake':
            self._stack.setCurrentIndex(1)
            self._ensure_cake()
        else:
            self._stack.setCurrentIndex(2)

    def _on_geom_toggled(self, checked):
        self._geom_body.setVisible(checked)
        self._geom_toggle.setText("▼  Geometry" if checked else "▶  Geometry")

    def _on_toggle_mask(self):
        self._mask_visible = not self._mask_visible
        self._btn_mask.setChecked(self._mask_visible)
        self._btn_mask.setText("Mask ●" if self._mask_visible else "Mask ○")
        self._raw_canvas.show_mask_overlay(self._mask, self._mask_visible)
        # Recompute cake with/without mask and update pixel-projected overlay
        if self._tth_lut is not None and self._image is not None:
            self._ensure_cake(reset_zoom=False)
        # Recompute lineout so it reflects the new mask state
        self._recompute_lineout()

    def _on_toggle_strain(self, checked):
        self._strain_visible = checked
        self._btn_strain.setText("Strain ●" if checked else "Strain ○")
        self._raw_canvas.set_strain_visible(checked)
        self._cake_canvas.set_strain_visible(checked)
        # Recompute cake and refresh the mask overlay (strain toggle changes
        # which cake bins are highlighted, so a fresh render is needed).
        if self._tth_lut is not None:
            self._ensure_cake()

    # ── Detector / Beam ───────────────────────────────────────────────────────

    def _on_detector_changed(self, idx):
        det = DETECTORS[DETECTOR_NAMES[idx]]
        self._nrows_ed.setText(str(det['nrows']))
        self._ncols_ed.setText(str(det['ncols']))
        self._px_ed.setText(str(det['px_um']))
        self._bcy_ed.setText(f"{det['ncols'] / 2.0:.1f}")
        self._bcz_ed.setText(f"{det['nrows'] / 2.0:.1f}")

    def _sync_energy_to_lambda(self):
        try:
            self._lambda_ed.setText(f"{_HC / float(self._energy_ed.text()):.7f}")
        except ValueError:
            pass

    def _sync_lambda_to_energy(self):
        try:
            self._energy_ed.setText(f"{_HC / float(self._lambda_ed.text()):.1f}")
        except ValueError:
            pass

    # ── Geometry ──────────────────────────────────────────────────────────────

    def _read_geom(self):
        return make_geometry(
            lsd=float(self._lsd_ed.text()),
            bc_y=float(self._bcy_ed.text()),
            bc_z=float(self._bcz_ed.text()),
            tx_deg=float(self._tx_ed.text()),
            ty_deg=float(self._ty_ed.text()),
            tz_deg=float(self._tz_ed.text()),
            px=float(self._px_ed.text()),
            nrows=int(self._nrows_ed.text()),
            ncols=int(self._ncols_ed.text()),
            wavelength=_HC / float(self._energy_ed.text()),
        )

    def _fill_geom_fields(self, geom):
        self._lsd_ed.setText(f"{geom['lsd']:.3f}")
        self._bcy_ed.setText(f"{geom['bc_y']:.3f}")
        self._bcz_ed.setText(f"{geom['bc_z']:.3f}")
        self._tx_ed.setText(f"{geom.get('tx_deg', 0.0):.6f}")
        self._ty_ed.setText(f"{geom.get('ty_deg', 0.0):.6f}")
        self._tz_ed.setText(f"{geom.get('tz_deg', 0.0):.6f}")
        # Sync nrows/ncols from geom so LUT dimensions match when no TIFF is loaded yet.
        # Also update detector dropdown if a preset matches.
        if 'nrows' in geom and 'ncols' in geom:
            nrows, ncols = int(geom['nrows']), int(geom['ncols'])
            self._nrows_ed.setText(str(nrows))
            self._ncols_ed.setText(str(ncols))
            for name in DETECTOR_NAMES:
                det = DETECTORS[name]
                if det['nrows'] == nrows and det['ncols'] == ncols:
                    idx = DETECTOR_NAMES.index(name)
                    self._det_combo.blockSignals(True)
                    self._det_combo.setCurrentIndex(idx)
                    self._det_combo.blockSignals(False)
                    break
        if 'px' in geom:
            self._px_ed.setText(str(geom['px']))

    # ── File actions ──────────────────────────────────────────────────────────

    def _on_load_tiff(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load calibration TIFF", "",
            "TIFF files (*.tif *.tiff);;All files (*)")
        if not path:
            return
        try:
            image = load_tiff(path)
            self._image = image
            self._image_path = path
            # Always sync nrows/ncols from the actual image so the LUT
            # dimensions match when calibration is run.
            nrows, ncols = image.shape[:2]
            self._nrows_ed.setText(str(nrows))
            self._ncols_ed.setText(str(ncols))
            self._raw_canvas.show_image(image)
            if self._mask_visible and self._mask is not None:
                self._raw_canvas.show_mask_overlay(self._mask, True)
            self._auto_refresh_rings()
        except Exception as e:
            log.error("Load image failed: %s", e, exc_info=True)
            QMessageBox.critical(self, "Error", str(e))

    def _load_toml(self, path):
        """Load geometry from *path* (.toml params file) into the GUI fields.

        Panel shifts embedded in [[panel_shifts]] tables are stored in
        self._panel_shifts and self._panel_map for immediate LUT building.

        Returns the full geometry dict from load_params().
        """
        from ..io import load_params
        geom = load_params(path)
        self._fill_geom_fields(geom)
        lam = geom.get('wavelength')
        if lam:
            self._lambda_ed.setText(f"{lam:.7f}")
            self._sync_lambda_to_energy()
        # Panel shifts are embedded in the .toml [[panel_shifts]] section
        ps = geom.get('panel_shifts', [])
        self._panel_shifts = ps if ps else None
        if self._panel_shifts:
            self._panel_map = make_panel_map(self._det_combo.currentText())
            log.info("Loaded %d panel shifts from %s",
                     len(self._panel_shifts), Path(path).name)
        self._auto_refresh_rings()
        return geom

    def _load_poni(self, path):
        """Load geometry from a pyFAI .poni file into the GUI fields.

        Uses the detector px and nrows currently set in the GUI to convert
        Poni1/Poni2 to bc_z/bc_y.  Panel shifts and distortion coefficients
        are not present in .poni files and are left unchanged.

        Returns the partial geometry dict produced by read_poni().
        """
        try:
            px    = float(self._px_ed.text())
            nrows = int(self._nrows_ed.text())
            ncols = int(self._ncols_ed.text())
        except ValueError:
            raise ValueError("Set detector pixel size and row count before loading a .poni file.")
        geom = read_poni(path, px, nrows)
        geom['nrows'] = nrows
        geom['ncols'] = ncols
        # .poni has no distortion or panel data — supply zero defaults
        for k in ('p0', 'p1', 'p2', 'p3', 'p4'):
            geom.setdefault(k, 0.0)
        geom.setdefault('rho_d', 217578.0)
        geom.setdefault('panel_shifts', [])
        self._fill_geom_fields(geom)
        self._lambda_ed.setText(f"{geom['wavelength']:.7f}")
        self._sync_lambda_to_energy()
        self._auto_refresh_rings()
        return geom

    def _load_midas_params(self, path):
        """Load geometry from a MIDAS geometry_params.txt into the GUI fields.

        Delegates to :func:`~midas4pil.io.load_midas_params`, which handles
        key aliases, BC_Z frame conversion (ImTransOpt 2), and the optional
        PanelShiftsFile.  Returns the complete geometry dict.
        """
        from ..io import load_midas_params
        geom = load_midas_params(path)
        self._fill_geom_fields(geom)
        self._lambda_ed.setText(f"{geom['wavelength']:.7f}")
        self._sync_lambda_to_energy()
        self._auto_refresh_rings()
        return geom

    def _on_open_mask_editor(self):
        if self._image is None:
            QMessageBox.information(self, "Mask Editor", "Load a TIFF first.")
            return
        from .mask_editor import MaskEditorDialog
        dlg = MaskEditorDialog(self._image, mask=self._mask,
                               panel_map=self._panel_map,
                               colors=self._mpl_colors, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_mask = dlg.result_mask()
            if new_mask is not None:
                self._mask = new_mask
                self._mask_visible = True
                self._btn_mask.setChecked(True)
                self._btn_mask.setText("Mask ●")
                self._raw_canvas.show_mask_overlay(self._mask, True)
                self._recompute_lineout()

    def _auto_refresh_rings(self):
        """Reload calibrant rings and redraw — called automatically on any
        change to calibrant material, wavelength, or 2θ max."""
        try:
            cal_name = self._cal_combo.currentText()
            tth_max  = float(self._tth_max_ed.text())
            lam      = float(self._lambda_ed.text())
            self._rings = [r for r in load_calibrant(cal_name, lam)
                           if r['tth'] <= tth_max]
        except Exception as e:
            log.warning("calibrant field parse failed: %s", e)
            return
        if self._image is not None:
            try:
                geom = self._read_geom()
                self._draw_rings(geom)
            except Exception as e:
                log.debug("ring draw failed: %s", e)

    def _draw_rings(self, geom):
        if self._image is None or not self._rings:
            return
        nrows = int(self._nrows_ed.text())
        ncols = int(self._ncols_ed.text())
        px    = float(self._px_ed.text())
        self._raw_canvas.draw_rings(
            self._rings, geom['bc_y'], geom['bc_z'],
            geom['lsd'], px, nrows, ncols,
            ty_deg=geom.get('ty_deg', 0.0),
            tz_deg=geom.get('tz_deg', 0.0))

    def _on_find_center_auto(self):
        """Sharpness-based beam-center search (no calibrant required)."""
        if self._image is None:
            QMessageBox.information(self, "Auto Find Center", "Load an image first.")
            return
        if not self._lsd_ed.text().strip():
            QMessageBox.warning(self, "Missing field",
                                "Please enter Lsd before finding the beam center.")
            return
        try:
            geom = self._read_geom()
            rough_max = float(self._rough_max_ed.text())
        except Exception as e:
            log.error("Auto find center setup failed: %s", e, exc_info=True)
            QMessageBox.critical(self, "Error", str(e))
            return
        self._start_find_center(geom, auto=True, rough_max_deg=rough_max)

    def _start_find_center(self, geom, auto, rough_max_deg=None):
        self._fc_worker = _FindCenterWorker(
            self._image, self._mask, self._rings, geom, auto=auto,
            rough_max_deg=rough_max_deg, parent=self)
        self._fc_worker.finished.connect(self._on_find_center_done)
        self._fc_worker.error.connect(self._on_find_center_error)
        self._btn_manual_center.setEnabled(False)
        self._btn_find_center_auto.setEnabled(False)
        label = "Auto find center running…" if auto else "Finding beam center…"
        self._results_lbl.setText(label)
        self._update_status_bar(label)
        mw = self.window()
        if hasattr(mw, 'show_busy'):
            mw.show_busy()
        from PySide6.QtWidgets import QApplication
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._fc_worker.start()

    def _on_find_center_done(self, bc_y, bc_z, lsd, ty_deg, tz_deg):
        from PySide6.QtWidgets import QApplication
        QApplication.restoreOverrideCursor()
        self._btn_manual_center.setEnabled(True)
        self._btn_find_center_auto.setEnabled(True)
        mw = self.window()
        if hasattr(mw, 'hide_progress'):
            mw.hide_progress()
        self._bcy_ed.setText(f"{bc_y:.3f}")
        self._bcz_ed.setText(f"{bc_z:.3f}")
        self._lsd_ed.setText(f"{lsd:.1f}")
        self._ty_ed.setText(f"{ty_deg:.6f}")
        self._tz_ed.setText(f"{tz_deg:.6f}")
        msg = (f"Center: BC_Y={bc_y:.3f}  BC_Z={bc_z:.3f}  Lsd={lsd:.0f} µm"
               f"  ty={ty_deg:.4f}°  tz={tz_deg:.4f}°")
        self._results_lbl.setText(msg)
        self._update_status_bar(msg)
        # Clear any stale ring overlay (circles or LUT) before updating.
        self._raw_canvas.clear_rings()
        if self._rings:
            try:
                geom = self._read_geom()
                # If a calibrated LUT already exists, rebuild it with the
                # updated beam centre / Lsd so the pixel-accurate overlay
                # remains consistent rather than falling back to flat circles.
                if self._tth_lut is not None:
                    nrows = int(self._nrows_ed.text())
                    ncols = int(self._ncols_ed.text())
                    _LUT_KEYS = ('bc_y', 'bc_z', 'lsd', 'px',
                                 'tx_deg', 'ty_deg', 'tz_deg',
                                 'p0', 'p1', 'p2', 'p3', 'p4', 'rho_d')
                    lut_kw = {k: geom[k] for k in _LUT_KEYS if k in geom}
                    if self._panel_shifts and self._panel_map is not None:
                        from ..panels import build_lut_with_panels
                        self._tth_lut, self._eta_lut = build_lut_with_panels(
                            nrows, ncols, **lut_kw,
                            panel_map=self._panel_map,
                            panel_shifts=self._panel_shifts)
                    else:
                        self._tth_lut, self._eta_lut = build_lut(
                            nrows, ncols, **lut_kw)
                    self._geom = geom
                    self._draw_rings_lut()
                else:
                    self._draw_rings(geom)
            except Exception as e:
                log.debug("Ring refresh failed: %s", e)

    def _on_find_center_error(self, msg):
        from PySide6.QtWidgets import QApplication
        QApplication.restoreOverrideCursor()
        self._btn_manual_center.setEnabled(True)
        self._btn_find_center_auto.setEnabled(True)
        mw = self.window()
        if hasattr(mw, 'hide_progress'):
            mw.hide_progress()
        err = f"Find center error: {msg}"
        self._results_lbl.setText(err)
        self._update_status_bar(err)
        QMessageBox.critical(self, "Find Center Error", msg)

    # ── Manual Find Center ────────────────────────────────────────────────────

    def _on_manual_center_start(self):
        """Enter interactive beam-center picking mode.

        Workflow
        --------
        1. Guard checks (image + calibrant rings must be loaded).
        2. Modal _RingPickDialog — user selects exactly which hkl ring they
           can see on the detector.  This removes all ring-assignment ambiguity.
        3. Enter point-picking mode: user clicks ≥ 3 points on that arc.
        4. On Done → circle fit → BC_Y, BC_Z, and Lsd are all determined
           from the single known ring 2θ without any guessing.
        """
        if self._image is None:
            QMessageBox.information(self, "Manual Find Center",
                                    "Load an image first.")
            return
        if not self._rings:
            QMessageBox.information(self, "Manual Find Center",
                                    "No calibrant rings loaded.\n"
                                    "Select a calibrant from the dropdown and "
                                    "set the X-ray energy first.")
            return

        # ── Step 1: ring selection (modal) ────────────────────────────────────
        pick_dlg = _RingPickDialog(self._rings, parent=self)
        if pick_dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._manual_center_ring = pick_dlg.selected_ring()
        if self._manual_center_ring is None:
            return

        # ── Step 2: enter point-picking mode ──────────────────────────────────
        self._manual_center_mode = True
        self._center_points.clear()
        for m in self._center_markers:
            try:
                m.remove()
            except Exception:
                pass
        self._center_markers.clear()
        self._raw_canvas.canvas.draw_idle()

        # Deactivate any active matplotlib toolbar tool (zoom / pan) so that
        # left-clicks on the canvas reach our custom interceptor rather than
        # being consumed by the toolbar's zoom-rect or pan handler.
        toolbar = getattr(self._raw_canvas.canvas, 'toolbar', None)
        if toolbar is not None and toolbar.mode:
            mode_str = str(toolbar.mode).lower()
            if 'zoom' in mode_str:
                toolbar.zoom()   # toggle zoom off
            elif 'pan' in mode_str:
                toolbar.pan()    # toggle pan off

        # Activate click interceptor on raw canvas
        self._raw_canvas._click_interceptor = self._on_manual_center_click

        # Disable other action buttons while picking
        self._btn_manual_center.setEnabled(False)
        self._btn_find_center_auto.setEnabled(False)
        self._btn_calibrate.setEnabled(False)

        # Switch to raw view and set crosshair cursor
        self._set_view('raw')
        from PySide6.QtWidgets import QApplication
        QApplication.setOverrideCursor(Qt.CursorShape.CrossCursor)

        # Open modeless instruction dialog (shows the chosen ring prominently)
        # Use lambda wrappers: PySide6 clicked(bool) → 0-arg slot can silently
        # drop the call on some Qt builds if the signature doesn't match exactly.
        self._mc_dialog = _ManualCenterDialog(
            ring=self._manual_center_ring, parent=self)
        self._mc_dialog.done_btn.clicked.connect(
            lambda: self._on_manual_center_done())
        self._mc_dialog.cancel_btn.clicked.connect(
            lambda: self._on_manual_center_cancel())
        # X button / Escape closes dialog → treat as cancel
        self._mc_dialog.rejected.connect(
            lambda: self._on_manual_center_cancel())
        self._mc_dialog.show()

        ring = self._manual_center_ring
        hkl  = f"({ring['h']} {ring['k']} {ring['l']})"
        self._update_status_bar(
            f"Manual Find Center [{hkl} 2θ={ring['tth']:.3f} deg]: "
            f"click \u22653 points on the arc, then Done.")

    def _on_manual_center_click(self, col, row):
        """Receive one click from _ZoomCanvas when in point-picking mode."""
        self._center_points.append((col, row))
        n = len(self._center_points)
        mk, = self._raw_canvas.ax.plot(
            col, row, '+', color='#ff9800', ms=10, mew=1.5, zorder=10)
        self._center_markers.append(mk)
        self._raw_canvas.canvas.draw_idle()
        if self._mc_dialog is not None:
            self._mc_dialog.update_count(n)
        suffix = "Click Done or add more." if n >= 3 else f"Need {3 - n} more."
        self._update_status_bar(
            f"Manual Find Center: {n} point{'s' if n != 1 else ''} selected. {suffix}")

    def _on_manual_center_done(self):
        """Fit clicked arc points to known ring tth using full tilt-aware geometry.

        Uses the current ty/tz from the UI fields (fixed), and optimises
        BC_Y, BC_Z, Lsd so that each clicked pixel maps to the selected ring's
        2θ via pixel_to_r_eta().  Works correctly for tilted detectors where
        rings appear as ellipses.
        """
        n = len(self._center_points)
        log.info("Manual center Done called — %d points", n)
        if n < 3:
            log.warning("Manual center Done: fewer than 3 points (%d), aborting", n)
            return
        try:
            from scipy.optimize import minimize
            from ..geometry import pixel_to_r_eta, build_tilt_matrix

            pts  = np.array(self._center_points, dtype=float)
            cols_p = pts[:, 0]   # column (x) coords
            rows_p = pts[:, 1]   # row    (y) coords

            ring = self._manual_center_ring
            hkl  = f"({ring['h']} {ring['k']} {ring['l']})" if ring else "?"
            tth_target = ring['tth']   # degrees

            # Read current geometry — ty/tz are held fixed during this fit
            geom   = self._read_geom()
            px     = geom['px']
            ty_deg = geom.get('ty_deg', 0.0)
            tz_deg = geom.get('tz_deg', 0.0)
            rho_d  = geom.get('rho_d', 217578.0)
            TRs    = build_tilt_matrix(0.0, ty_deg, tz_deg)

            def _obj(params):
                bc_y, bc_z, lsd_um = params
                if lsd_um <= 0:
                    return 1e12
                R_px, _ = pixel_to_r_eta(
                    cols_p, rows_p, bc_y, bc_z, TRs, lsd_um,
                    rho_d, 0.0, 0.0, 0.0, 0.0, 0.0, px)
                tth_comp = np.degrees(np.arctan(R_px * px / lsd_um))
                return float(np.sum((tth_comp - tth_target) ** 2))

            x0  = [geom['bc_y'], geom['bc_z'], geom['lsd']]
            res = minimize(_obj, x0, method='Nelder-Mead',
                           options={'xatol': 0.05, 'fatol': 1e-6,
                                    'maxiter': 5000, 'adaptive': True})

            bc_y, bc_z, lsd = float(res.x[0]), float(res.x[1]), float(res.x[2])
            log.info("Manual center fit: BC_Y=%.3f  BC_Z=%.3f  Lsd=%.1f um  "
                     "nfev=%d  rms_err=%.4f deg",
                     bc_y, bc_z, lsd, res.nfev,
                     np.sqrt(res.fun / n))

            self._bcy_ed.setText(f"{bc_y:.3f}")
            self._bcz_ed.setText(f"{bc_z:.3f}")
            self._lsd_ed.setText(f"{lsd:.1f}")

            msg = (f"Manual center: BC_Y={bc_y:.3f}  BC_Z={bc_z:.3f}"
                   f"  Lsd={lsd:.0f} µm"
                   f"  ({n} pts, ring {hkl} 2θ={tth_target:.3f}°)")
            self._results_lbl.setText(msg)
            self._update_status_bar(msg)
            self._auto_refresh_rings()
        except Exception as e:
            log.error("Manual center fit failed: %s", e, exc_info=True)
            QMessageBox.critical(self, "Fit Error", str(e))
        finally:
            self._exit_manual_center_mode()

    def _on_manual_center_cancel(self):
        if not self._manual_center_mode:
            return   # already exited (e.g. dialog close triggered after _exit)
        self._exit_manual_center_mode()
        self._update_status_bar("Manual Find Center cancelled.")

    def _exit_manual_center_mode(self):
        """Clean up after manual center picking (done or cancelled)."""
        from PySide6.QtWidgets import QApplication
        # Clear mode flag first — prevents re-entry when dialog.close() emits rejected
        self._manual_center_mode = False
        self._center_points.clear()
        for m in self._center_markers:
            try:
                m.remove()
            except Exception:
                pass
        self._center_markers.clear()
        self._raw_canvas.canvas.draw_idle()
        self._raw_canvas._click_interceptor = None
        QApplication.restoreOverrideCursor()
        if self._mc_dialog is not None:
            dlg, self._mc_dialog = self._mc_dialog, None
            dlg.close()
        self._btn_manual_center.setEnabled(True)
        self._btn_find_center_auto.setEnabled(True)
        self._btn_calibrate.setEnabled(True)

    # ── Calibration ───────────────────────────────────────────────────────────

    def _on_calibrate(self):
        if self._image is None:
            QMessageBox.information(self, "Calibrate", "Load a TIFF first.")
            return
        # Validate required fields before attempting geometry read
        missing = []
        if not self._lsd_ed.text().strip():
            missing.append("Lsd (sample–detector distance)")
        if not self._energy_ed.text().strip() and not self._lambda_ed.text().strip():
            missing.append("Energy (eV) or Wavelength (Å)")
        if missing:
            QMessageBox.warning(self, "Missing fields",
                                "Please fill in:\n  • " + "\n  • ".join(missing))
            return
        try:
            geom      = self._read_geom()
            cal_name  = self._cal_combo.currentText()
            tth_max   = float(self._tth_max_ed.text())
            rough_max = float(self._rough_max_ed.text())
            lam       = float(self._lambda_ed.text())
            rings_all = [r for r in load_calibrant(cal_name, lam)
                         if r['tth'] <= tth_max]
            panel_map  = make_panel_map(self._det_combo.currentText()) if self._chk_panels.isChecked() else None
            n_stages   = 3 if self._chk_panels.isChecked() else 2
            do_panels  = self._chk_panels.isChecked()
            do_dist    = self._chk_distortion.isChecked()
            both       = do_panels and do_dist
            # Stage descriptions for _on_stage_done progress hints.
            # Order: panels always before distortion (panels correct pixel
            # coordinates that the distortion polynomial is defined on top of).
            if both:
                self._stage_next_desc = {1: 'panel shifts',
                                         2: 'panel shifts + distortion'}
            elif do_panels:
                self._stage_next_desc = {1: 'full calibration',
                                         2: 'panel shifts'}
            else:
                self._stage_next_desc = {1: 'full calibration'}

            log.info("Starting calibration: %d rings, %d stages, panels=%s, distortion=%s",
                     len(rings_all), n_stages, do_panels, do_dist)

            self._panel_map = panel_map          # store for LUT building in _on_calib_finished
            self._calib_worker = _CalibWorker(
                self._image, self._mask, geom, panel_map, rings_all,
                rough_max, self._chk_panels.isChecked(),
                fit_distortion=self._chk_distortion.isChecked(),
                parent=self)
            self._calib_worker.stage_done.connect(self._on_stage_done)
            self._calib_worker.finished.connect(self._on_calib_finished)
            self._calib_worker.error.connect(self._on_calib_error)
            self._calib_worker.status_msg.connect(self._on_calib_status)
            self._calib_worker.progress.connect(self._on_calib_progress)
            self._btn_calibrate.setEnabled(False)
            self._btn_stop.setEnabled(True)
            self._n_stages = n_stages
            init_msg = f"Stage 1/{n_stages}: rough alignment…"
            self._results_lbl.setText(init_msg)
            self._update_status_bar(init_msg)
            from PySide6.QtWidgets import QApplication
            from PySide6.QtCore import Qt
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self._calib_worker.start()
        except Exception as e:
            log.warning("Calibration setup error: %s", e, exc_info=True)
            QMessageBox.critical(self, "Calibration setup error", str(e))

    def _on_calib_progress(self, completed, total):
        mw = self.window()
        if hasattr(mw, 'show_progress'):
            mw.show_progress(completed, total)

    def _on_stop(self):
        if self._calib_worker and self._calib_worker.isRunning():
            # Cooperative cancellation: the worker checks isInterruptionRequested()
            # in its progress callback and raises _CalibrationCancelled to exit
            # cleanly after the current scipy iteration finishes.
            # Never call terminate() — on Linux it sends SIGTERM to the thread
            # mid-computation and deadlocks the main thread on GIL / Qt state.
            self._calib_worker.requestInterruption()
        from PySide6.QtWidgets import QApplication
        QApplication.restoreOverrideCursor()
        mw = self.window()
        if hasattr(mw, 'hide_progress'):
            mw.hide_progress()
        self._btn_calibrate.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._results_lbl.setText("Stopping…")
        self._update_status_bar("Calibration stopping…")

    def _on_reset(self):
        """Stop any running calibration and reset all state."""
        from PySide6.QtWidgets import QApplication
        if self._calib_worker and self._calib_worker.isRunning():
            self._calib_worker.requestInterruption()  # cooperative; no terminate()
            QApplication.restoreOverrideCursor()
        if self._fc_worker and self._fc_worker.isRunning():
            self._fc_worker.requestInterruption()
            QApplication.restoreOverrideCursor()
        mw = self.window()
        if hasattr(mw, 'hide_progress'):
            mw.hide_progress()

        self._image         = None
        self._image_path    = None
        self._mask          = None
        self._rings         = []
        self._geom          = None
        self._calib_result  = None
        self._tth_lut       = None
        self._eta_lut       = None
        self._cake_tth      = None
        self._cake_eta      = None
        self._mask_visible  = False
        self._calib_worker  = None
        self._lineout       = None
        self._lo_show       = {'I': True, 'I_sub': False, 'SNIP': False}
        self._lo_show_sigma = True
        self._btn_lo_sigma.setChecked(True)
        self._cake_ring_lines = []
        self._panel_map    = None
        self._panel_shifts = None

        self._btn_calibrate.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._btn_mask.setChecked(False)
        self._btn_mask.setText("Mask \u25cb")
        self._btn_lo_I.setChecked(True)
        self._btn_lo_Isub.setChecked(False)
        self._btn_lo_snip.setChecked(False)
        self._lo_pos_label.setText("  —  ")

        self._results_lbl.setText("Reset — load an image to begin.")
        self._update_status_bar("Calibration reset.")

        # Clear all display panels
        self._raw_canvas.clear()
        self._cake_canvas.clear()
        self.line_ax.clear()
        self._style_lineout_ax()
        self.line_ax.set_xlabel('2\u03b8 (deg)', fontsize=8)
        self.line_ax.set_ylabel('Intensity', fontsize=8)
        self.line_ax.set_title('Lineout', fontsize=9)
        self.line_canvas.draw_idle()
        log.info("Calibration reset by user.")

    def _update_status_bar(self, msg):
        """Push *msg* to the main window status bar if reachable."""
        try:
            self.window().statusBar().showMessage(msg)
        except Exception:
            pass

    def _on_calib_status(self, msg):
        """Live per-iteration progress from the calibration worker."""
        self._results_lbl.setText(msg)
        self._update_status_bar(msg)

    def _on_stage_done(self, stage, result):
        geom      = result['geom']
        strain    = result.get('mean_strain', float('nan')) * 1e6
        n_stages  = getattr(self, '_n_stages', 2)
        next_desc = getattr(self, '_stage_next_desc', {}).get(stage, 'finishing')
        self._fill_geom_fields(geom)
        done_txt = (f"Stage {stage}/{n_stages} done — "
                    f"Lsd={geom['lsd']:.1f} µm  "
                    f"BC=({geom['bc_y']:.2f}, {geom['bc_z']:.2f})  "
                    f"strain={strain:.0f} ppm")
        if stage < n_stages:
            status = f"{done_txt}   |   Stage {stage+1}/{n_stages}: {next_desc}…"
        else:
            status = done_txt
        self._results_lbl.setText(status)
        self._update_status_bar(status)
        if 'tth' in result and 'I' in result:
            self._show_lineout(result['tth'], result['I'],
                               result.get('bg'), result.get('I_sub'),
                               result.get('sigma'), result.get('px_cnt'))

    def _on_calib_finished(self, result):
        if (self._calib_worker is None or
                self._calib_worker.isInterruptionRequested()):
            return   # user stopped or reset — ignore late finish signal
        from PySide6.QtWidgets import QApplication
        QApplication.restoreOverrideCursor()
        mw = self.window()
        if hasattr(mw, 'hide_progress'):
            mw.hide_progress()
        self._btn_calibrate.setEnabled(True)
        self._btn_stop.setEnabled(False)
        geom   = result['geom']
        self._geom         = geom
        self._calib_result = result
        strain = result.get('mean_strain', float('nan')) * 1e6
        done_msg = (f"Done.  Lsd={geom['lsd']:.1f} µm  "
                    f"BC=({geom['bc_y']:.2f}, {geom['bc_z']:.2f})  "
                    f"strain={strain:.0f} ppm")
        self._results_lbl.setText(done_msg)
        self._update_status_bar(done_msg)
        log.info("Calibration finished — Lsd=%.1f µm  BC=(%.2f, %.2f)  "
                 "strain=%.0f ppm",
                 geom['lsd'], geom['bc_y'], geom['bc_z'], strain)
        try:
            nrows = int(self._nrows_ed.text())
            ncols = int(self._ncols_ed.text())
            px    = float(self._px_ed.text())
            _LUT_KEYS = ('bc_y', 'bc_z', 'lsd', 'px',
                         'tx_deg', 'ty_deg', 'tz_deg',
                         'p0', 'p1', 'p2', 'p3', 'p4', 'rho_d')
            lut_kw = {k: geom[k] for k in _LUT_KEYS if k in geom}
            panel_shifts = result.get('panel_shifts') or []
            self._panel_shifts = panel_shifts if panel_shifts else None
            geom['panel_shifts'] = panel_shifts
            if panel_shifts and self._panel_map is not None:
                from ..panels import build_lut_with_panels
                self._tth_lut, self._eta_lut = build_lut_with_panels(
                    nrows, ncols, **lut_kw,
                    panel_map=self._panel_map,
                    panel_shifts=panel_shifts)
                log.info("LUTs built with panel corrections (%d × %d)", nrows, ncols)
            else:
                self._tth_lut, self._eta_lut = build_lut(nrows, ncols, **lut_kw)
                log.info("LUTs built (%d × %d)", nrows, ncols)

            # Derive integration limits from actual detector coverage.
            # Pass px/lsd so the eta-resolution constraint is applied to tth_min.
            tth_min_lut, tth_max_lut = lut_tth_range(
                self._tth_lut, mask=self._mask,
                px=geom.get('px'), lsd=geom.get('lsd'))
            geom['tth_min'] = tth_min_lut
            geom['tth_max'] = tth_max_lut
            log.info("2th range from LUT: %.2f deg - %.2f deg", tth_min_lut, tth_max_lut)
        except Exception as e:
            log.error("build_lut failed: %s", e, exc_info=True)
            return

        # Replace circular ring arcs with pixel-accurate LUT overlay
        if self._rings:
            self._draw_rings_lut()

        # Compute and display lineout respecting the current mask toggle state.
        self._recompute_lineout()

    def _on_calib_error(self, msg):
        if (self._calib_worker is None or
                self._calib_worker.isInterruptionRequested()):
            return   # user stopped or reset — ignore late error signal
        from PySide6.QtWidgets import QApplication
        QApplication.restoreOverrideCursor()
        mw = self.window()
        if hasattr(mw, 'hide_progress'):
            mw.hide_progress()
        self._btn_calibrate.setEnabled(True)
        self._btn_stop.setEnabled(False)
        err_msg = f"Error: {msg}"
        self._results_lbl.setText(err_msg)
        self._update_status_bar(err_msg)
        log.error("Calibration error: %s", msg)
        QMessageBox.critical(self, "Calibration error", msg)

    # ── Display helpers ───────────────────────────────────────────────────────

    def _draw_rings_lut(self):
        """Replace circular ring arcs with strain-colored LUT overlay."""
        if self._tth_lut is None or not self._rings:
            return
        # Compute 2θ acceptance half-window from actual geometry (same formula as
        # the optimizer): one pixel subtends px/Lsd radians; the cos² factor
        # accounts for the foreshortening of pixels at high 2θ.
        px_um = float(self._px_ed.text())
        lsd_um = self._geom['lsd']
        median_tth = np.median([r['tth'] for r in self._rings])
        tth_tol = (np.degrees(px_um / lsd_um
                               * np.cos(np.radians(median_tth)) ** 2)
                   * _TTH_TOL_FACTOR)
        # Remove both circular-arc rings and any prior LUT overlay first
        self._raw_canvas.clear_rings()
        vmin, vmax = self._raw_canvas.show_lut_ring_overlay(
            self._tth_lut, self._rings, tth_tol,
            mask=self._mask,
            visible=self._strain_visible)
        if np.isfinite(vmax):
            cur = self._results_lbl.text()
            self._results_lbl.setText(
                cur + f"\nRing strain range: ±{vmax:.0f} ppm"
                      f"  (blue=expanded, red=compressed)")

    def _ensure_cake(self, reset_zoom=True):
        """Compute and show caked image if geometry and image are available.

        Applies the mask only when _mask_visible is True (consistent with the
        integration tab where _use_mask controls the same behaviour).
        Uses outer-bin-edge extent so overlay alignment matches display_widget.
        """
        if self._image is None:
            return
        if self._tth_lut is None or self._eta_lut is None:
            self._results_lbl.setText("Run calibration first to view cake.")
            return
        # Apply mask only when the mask button is ON (consistent with integ tab)
        effective_mask = self._mask if self._mask_visible else None
        cake, tth_c, eta_c, counts_cake = _compute_cake(
            self._image, effective_mask, self._tth_lut, self._eta_lut)
        if cake is None:
            return
        self._cake_tth = tth_c
        self._cake_eta = eta_c
        # Outer-bin-edge extent (consistent with display_widget)
        n_tth, n_eta = len(tth_c), len(eta_c)
        tth_l = float(tth_c[0]  - 0.5*(tth_c[1]  - tth_c[0])  if n_tth > 1 else tth_c[0])
        tth_r = float(tth_c[-1] + 0.5*(tth_c[-1] - tth_c[-2]) if n_tth > 1 else tth_c[-1])
        eta_b = float(eta_c[0]  - 0.5*(eta_c[1]  - eta_c[0])  if n_eta > 1 else eta_c[0])
        eta_t = float(eta_c[-1] + 0.5*(eta_c[-1] - eta_c[-2]) if n_eta > 1 else eta_c[-1])
        ext = [tth_l, tth_r, eta_b, eta_t]
        self._cake_canvas.show_image(
            cake, extent=ext,
            xlabel='2θ (deg)', ylabel='η (deg)')
        # Pixel-projected mask overlay (consistent with display_widget)
        if self._mask_visible and self._mask is not None:
            rgba, ext_ov = _compute_cake_mask_rgba(
                self._mask, self._tth_lut, self._eta_lut, tth_c, eta_c)
            self._cake_canvas.show_cake_mask_overlay(rgba, ext_ov or ext)
        else:
            self._cake_canvas.show_cake_mask_overlay(None, ext, visible=False)
        # Draw calibrant ring positions as vertical lines on cake (x-axis is 2θ)
        for ln in self._cake_ring_lines:
            try:
                ln.remove()
            except Exception:
                pass
        self._cake_ring_lines = []
        for ring in self._rings:
            ln = self._cake_canvas.ax.axvline(
                ring['tth'], color=_XH2, lw=0.8, ls='--', alpha=0.65)
            self._cake_ring_lines.append(ln)
        if self._cake_ring_lines:
            self._cake_canvas.canvas.draw_idle()

        # Strain overlay on cake — requires calibrated LUT and at least one ring
        if self._tth_lut is not None and self._rings and self._geom is not None:
            try:
                px_um  = float(self._px_ed.text())
                lsd_um = self._geom['lsd']
                median_tth = np.median([r['tth'] for r in self._rings])
                tth_tol = (np.degrees(px_um / lsd_um
                                      * np.cos(np.radians(median_tth)) ** 2)
                           * _TTH_TOL_FACTOR)
                self._cake_canvas.show_cake_strain_overlay(
                    tth_c, self._rings, tth_tol, ext,
                    px_cnt_cake=counts_cake,
                    visible=self._strain_visible)
            except Exception as e:
                log.debug("Cake strain overlay failed: %s", e)

        if reset_zoom:
            self._cake_canvas._reset_zoom()

    def _recompute_lineout(self):
        """Recompute the 1-D lineout using the current mask state and show it.

        Called whenever the mask is toggled or calibration finishes.
        Uses the mask only when _mask_visible is True; ignores it (all pixels
        active) otherwise.  Does nothing if geometry or image is not ready.
        """
        if self._tth_lut is None or self._image is None or self._geom is None:
            return
        mask = (self._mask
                if (self._mask is not None and self._mask_visible)
                else np.zeros(self._image.shape, dtype=bool))
        try:
            mode = self._geom.get('mode', 'varbin')
            if mode == 'varbin':
                tth, I, bg, I_sub, sigma, px_cnt = integrate_1d_varbin(
                    self._image, mask, self._tth_lut,
                    self._geom['tth_min'],
                    self._geom['tth_max'],
                    self._geom['px'],
                    self._geom['lsd'],
                    eta_lut=self._eta_lut)
            else:
                tth, I, bg, I_sub, sigma, px_cnt = integrate_1d(
                    self._image, mask, self._tth_lut,
                    self._geom['tth_min'],
                    self._geom['tth_max'],
                    self._geom.get('tth_bin_size', 0.025),
                    eta_lut=self._eta_lut)
            self._show_lineout(tth, I, bg, I_sub, sigma, px_cnt)
        except Exception as e:
            log.warning("Lineout recomputation failed: %s", e)

    def _show_lineout(self, tth, I, bg=None, I_sub=None, sigma=None, px_cnt=None):
        self._lineout = (tth, I, bg, I_sub, sigma, px_cnt)
        self._refresh_lineout()

    def _toggle_lo_curve(self, name, checked):
        self._lo_show[name] = checked
        self._refresh_lineout()

    def _toggle_lo_sigma(self, checked):
        self._lo_show_sigma = checked
        self._refresh_lineout()

    def _style_lineout_ax(self):
        """Apply current theme colours to line_ax (replaces hardcoded _style_ax)."""
        c = self._mpl_colors
        self.line_ax.set_facecolor(c.get('ax', _BG_AX))
        self.line_ax.tick_params(colors=c.get('fg', _FG), labelsize=7)
        for sp in self.line_ax.spines.values():
            sp.set_color(c.get('spine', '#999999'))
        self.line_ax.xaxis.label.set_color(c.get('fg', _FG))
        self.line_ax.yaxis.label.set_color(c.get('fg', _FG))
        self.line_ax.title.set_color(c.get('fg', _FG))

    def _refresh_lineout(self):
        """Redraw the lineout — any combination of I / I_sub / SNIP can be shown."""
        if self._lineout is None:
            return
        tth, I, bg, I_sub, sigma, px_cnt = self._lineout
        c = self._mpl_colors

        curves = []
        if self._lo_show.get('I', True):
            curves.append((tth, I, c.get('line_I', '#64b5f6'), 'I'))
        if self._lo_show.get('I_sub', False) and I_sub is not None:
            curves.append((tth, I_sub, c.get('line_Isub', _XH), 'I\u208b'))
        if self._lo_show.get('SNIP', False) and bg is not None:
            curves.append((tth, bg, c.get('line_snip', '#ff9800'), 'SNIP bg'))

        self.line_ax.clear()
        self._style_lineout_ax()
        for ring in self._rings:
            self.line_ax.axvline(ring['tth'], color=c.get('ring', _XH2),
                                 lw=0.8, ls='--', alpha=0.65)
        if curves:
            for x, y, color, label in curves:
                self.line_ax.plot(x, y, color=color, lw=0.8, label=label)
                if self._lo_show_sigma and sigma is not None and label != 'SNIP bg':
                    self.line_ax.fill_between(
                        x, y - sigma, y + sigma,
                        alpha=0.18, color=color, linewidth=0)
            if len(curves) >= 2:
                self.line_ax.legend(fontsize=7, facecolor=c.get('ax', _BG_AX),
                                    labelcolor=c.get('fg', _FG), framealpha=0.7)
            title = 'Lineout \u2014 ' + ' + '.join(lbl for _, _, _, lbl in curves)
        else:
            title = 'Lineout'
        self.line_ax.set_xlabel('2\u03b8 (deg)', fontsize=8)
        self.line_ax.set_ylabel('Intensity', fontsize=8)
        self.line_ax.set_title(title, fontsize=9)
        self._lo_home_xlim = tuple(self.line_ax.get_xlim())
        self._lo_home_ylim = tuple(self.line_ax.get_ylim())
        self._lo_zoom_stack.clear()
        self.line_canvas.draw_idle()

    # ── Mouse hover (position readout) ────────────────────────────────────────

    def _on_raw_motion(self, event):
        if event.inaxes != self._raw_canvas.ax or event.xdata is None:
            return
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        I_str = ""
        if (self._image is not None
                and 0 <= row < self._image.shape[0]
                and 0 <= col < self._image.shape[1]):
            I_str = f"   I = {self._image[row, col]:.0f}"
        if (self._tth_lut is not None
                and 0 <= row < self._tth_lut.shape[0]
                and 0 <= col < self._tth_lut.shape[1]):
            tth = self._tth_lut[row, col]
            eta = self._eta_lut[row, col]
            self._pos_label.setText(
                f"  2θ = {tth:.3f} deg   η = {eta:.1f} deg{I_str}   px ({col}, {row})")
        else:
            self._pos_label.setText(f"  px ({col}, {row}){I_str}")

    def _on_cake_motion(self, event):
        if event.inaxes != self._cake_canvas.ax or event.xdata is None:
            return
        tth, eta = event.xdata, event.ydata
        I_str = ""
        cake_img = self._cake_canvas._image
        if cake_img is not None and hasattr(self, '_tth_lut') and self._tth_lut is not None:
            # cake_img shape is (n_eta, n_tth); extent = [tth0, tth1, eta0, eta1]
            xlim = self._cake_canvas._home_xlim
            ylim = self._cake_canvas._home_ylim
            if xlim is not None:
                n_eta, n_tth = cake_img.shape
                ti = int(np.clip(
                    round((tth - xlim[0]) / (xlim[1] - xlim[0]) * (n_tth - 1)),
                    0, n_tth - 1))
                ei = int(np.clip(
                    round((eta - ylim[0]) / (ylim[1] - ylim[0]) * (n_eta - 1)),
                    0, n_eta - 1))
                I_val = float(cake_img[ei, ti])
                if np.isfinite(I_val):
                    I_str = f"   I = {I_val:.0f}"
        self._pos_label.setText(
            f"  2θ = {tth:.3f} deg   η = {eta:.1f} deg{I_str}")

    def _on_lo_position(self, event):
        """Update 2θ / I position label while hovering over the lineout."""
        if event.inaxes != self.line_ax or event.xdata is None:
            self._lo_pos_label.setText("  —  ")
            return
        tth = event.xdata
        parts = []
        if self._lineout is not None:
            tth_arr, I_arr, bg_arr, I_sub_arr, sig_arr, px_cnt_arr = self._lineout
            if len(tth_arr) > 0:
                idx = int(np.argmin(np.abs(tth_arr - tth)))
                if self._lo_show.get('I', True) and 0 <= idx < len(I_arr):
                    parts.append(f"I={I_arr[idx]:.2f}")
                if (self._lo_show.get('I_sub', False) and I_sub_arr is not None
                        and 0 <= idx < len(I_sub_arr)):
                    parts.append(f"I\u208b={I_sub_arr[idx]:.2f}")
                if (self._lo_show.get('SNIP', False) and bg_arr is not None
                        and 0 <= idx < len(bg_arr)):
                    parts.append(f"SNIP={bg_arr[idx]:.2f}")
                if sig_arr is not None and 0 <= idx < len(sig_arr):
                    parts.append(f"\u03c3={sig_arr[idx]:.2f}")
                if px_cnt_arr is not None and 0 <= idx < len(px_cnt_arr):
                    parts.append(f"px_cnt={int(px_cnt_arr[idx])}")
        I_str = "   " + "  ".join(parts) if parts else ""
        self._lo_pos_label.setText(f"  2\u03b8 = {tth:.3f} deg{I_str}")

    # ── Lineout zoom ──────────────────────────────────────────────────────────

    def _lo_reset_zoom(self):
        if self._lo_home_xlim is not None:
            self.line_ax.set_xlim(self._lo_home_xlim)
            self.line_ax.set_ylim(self._lo_home_ylim)
        self._lo_zoom_stack.clear()
        self.line_canvas.draw_idle()

    def _on_lo_scroll(self, event):
        if event.inaxes != self.line_ax or event.xdata is None:
            return
        factor = 1.35
        scale  = 1.0 / factor if event.button == 'up' else factor
        xc, yc = event.xdata, event.ydata
        xlim = list(self.line_ax.get_xlim())
        ylim = list(self.line_ax.get_ylim())
        new_x = [xc + (v - xc) * scale for v in xlim]
        new_y = [yc + (v - yc) * scale for v in ylim]
        if event.button == 'down' and self._lo_home_xlim is not None:
            home_span = abs(self._lo_home_xlim[1] - self._lo_home_xlim[0])
            if abs(new_x[1] - new_x[0]) >= home_span:
                self._lo_reset_zoom()
                return
        self._lo_zoom_stack.append((xlim, ylim))
        self.line_ax.set_xlim(new_x)
        self.line_ax.set_ylim(new_y)
        self.line_canvas.draw_idle()

    def _on_lo_press(self, event):
        if event.inaxes != self.line_ax:
            return
        if event.button == 3:
            if self._lo_zoom_stack:
                xlim, ylim = self._lo_zoom_stack.pop()
                self.line_ax.set_xlim(xlim)
                self.line_ax.set_ylim(ylim)
                self.line_canvas.draw_idle()
            else:
                self._lo_reset_zoom()
            return
        if getattr(self.line_canvas, 'toolbar', None) and self.line_canvas.toolbar.mode:
            return
        if event.button == 1 and event.xdata is not None:
            self._lo_dragging     = True
            self._lo_roi_start    = (event.xdata, event.ydata)
            self._lo_roi_start_px = (event.x, event.y)

    def _on_lo_motion(self, event):
        if not self._lo_dragging or self._lo_roi_start is None:
            return
        if event.inaxes != self.line_ax or event.xdata is None:
            return
        x0, y0 = self._lo_roi_start
        dx, dy = event.xdata - x0, event.ydata - y0
        if self._lo_roi_patch is not None:
            self._lo_roi_patch.remove()
        from matplotlib.patches import Rectangle as _R
        self._lo_roi_patch = _R(
            (x0, y0), dx, dy,
            lw=1, edgecolor=_XH, facecolor='none', ls='--', alpha=0.85)
        self.line_ax.add_patch(self._lo_roi_patch)
        self.line_canvas.draw_idle()

    def _on_lo_release(self, event):
        if not self._lo_dragging or self._lo_roi_start is None:
            return
        self._lo_dragging = False
        if self._lo_roi_patch is not None:
            try:
                self._lo_roi_patch.remove()
            except Exception:
                pass
            self._lo_roi_patch = None
        if event.inaxes != self.line_ax or event.xdata is None:
            self._lo_roi_start = None
            self.line_canvas.draw_idle()
            return
        x0, y0 = self._lo_roi_start
        self._lo_roi_start = None
        dx = event.xdata - x0
        dy = event.ydata - y0
        px_start = self._lo_roi_start_px
        self._lo_roi_start_px = None
        if px_start is not None:
            if abs(event.x - px_start[0]) < 5:
                self.line_canvas.draw_idle()
                return
        elif abs(dx) < 1e-9:
            self.line_canvas.draw_idle()
            return
        self._lo_zoom_stack.append(
            (list(self.line_ax.get_xlim()), list(self.line_ax.get_ylim())))
        x0n, x1n = sorted([x0, x0 + dx])
        self.line_ax.set_xlim(x0n, x1n)
        if abs(dy) > 1e-9:
            y0n, y1n = sorted([y0, y0 + dy])
            self.line_ax.set_ylim(y0n, y1n)
        self.line_canvas.draw_idle()

    # ── Result dialog ─────────────────────────────────────────────────────────

    def _on_show_result(self):
        from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                                        QPlainTextEdit, QPushButton, QLabel)
        dlg = QDialog(self)
        dlg.setWindowTitle("Calibration Result")
        dlg.resize(620, 680)
        vl = QVBoxLayout(dlg)

        if self._calib_result is None:
            vl.addWidget(QLabel("No calibration result yet.  Run calibration first."))
        else:
            txt = QPlainTextEdit(self._format_calib_result())
            txt.setReadOnly(True)
            txt.setFont(__import__('PySide6.QtGui', fromlist=['QFont']).QFont(
                "Monospace", 9))
            vl.addWidget(txt)

            hl = QHBoxLayout()
            btn_params  = QPushButton("Save Params…")
            btn_poni    = QPushButton("Export .poni…")
            btn_lineout = QPushButton("Save Lineout…")
            btn_cake    = QPushButton("Save Cake…")
            btn_midas   = QPushButton("Export MIDAS…")
            btn_params.clicked.connect(self._on_export_params)
            btn_poni.clicked.connect(self._on_export_poni)
            btn_midas.clicked.connect(self._on_export_midas_params)
            btn_lineout.clicked.connect(self._on_save_lineout)
            btn_cake.clicked.connect(self._on_save_cake)
            btn_poni.setToolTip("Export geometry as a .poni file")
            btn_midas.setToolTip("Export geometry as a MIDAS geometry_params.txt")
            hl.addWidget(btn_params)
            hl.addWidget(btn_poni)
            hl.addWidget(btn_midas)
            hl.addWidget(btn_lineout)
            hl.addWidget(btn_cake)
            hl.addStretch()
            vl.addLayout(hl)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        vl.addWidget(btn_close)
        dlg.exec()

    def _format_calib_result(self):
        r    = self._calib_result
        geom = r['geom']
        W    = 46   # ruler width
        lines = []

        # ── 1. Flat detector geometry (PONI) ──────────────────────────────────
        lines.append("1. Flat Detector Geometry (PONI)")
        lines.append("─" * W)
        lines.append("   Corrects position/orientation of the ideal flat detector")
        lines.append("   plane relative to the beam and sample (6 DOF, tx fixed).")
        lines.append(f"  {'Lsd':6s}  {geom['lsd']:>16.3f}  µm   (sample–detector distance)")
        lines.append(f"  {'BC_Y':6s}  {geom['bc_y']:>16.3f}  px   (beam centre, col from left)")
        lines.append(f"  {'BC_Z':6s}  {geom['bc_z']:>16.3f}  px   (beam centre, row from top)")
        lines.append(f"  {'tx':6s}  {geom.get('tx_deg', 0.0):>16.6f}  deg  (rotation around X — not fitted)")
        lines.append(f"  {'ty':6s}  {geom.get('ty_deg', 0.0):>16.6f}  deg  (rotation around Y)")
        lines.append(f"  {'tz':6s}  {geom.get('tz_deg', 0.0):>16.6f}  deg  (rotation around Z)")
        lines.append("")

        # ── 2. Panel shift correction ─────────────────────────────────────────
        # Listed before distortion: panel shifts correct pixel coordinates
        # (hardware geometry); distortion is a smooth residual defined on top.
        panel_shifts = r.get('panel_shifts')
        panel_coverage = r.get('panel_coverage', {})
        if panel_shifts:
            lines.append("2. Panel Shift Correction (per-module rigid-body)")
            lines.append("─" * W)
            lines.append("   Per-panel dY/dZ (px), dLsd (µm), dTheta (deg).")
            lines.append("   Hardware property — calibrate once, reuse indefinitely.")
            hdr2 = (f"  {'ID':>4}  {'dY(px)':>8}  {'dZ(px)':>8}"
                    f"  {'dLsd(µm)':>9}  {'dTheta(deg)':>11}  {'rings':>5}  {'pts':>6}")
            lines.append(hdr2)
            lines.append("  " + "─" * (len(hdr2) - 2))
            uncovered_ids = []
            for ps in panel_shifts:
                pid = ps['id']
                cov = panel_coverage.get(pid, {})
                n_r = cov.get('n_rings', 0)
                n_p = cov.get('n_pixels', 0)
                flag = "  *" if n_p == 0 else ""
                lines.append(
                    f"  {pid:4d}  {ps['dY']:8.3f}  {ps['dZ']:8.3f}"
                    f"  {ps['dLsd']:9.3f}  {ps['dTheta']:10.4f}"
                    f"  {n_r:5d}  {n_p:6d}{flag}")
                if n_p == 0:
                    uncovered_ids.append(pid)
            if uncovered_ids:
                lines.append("")
                lines.append(f"  * Panel(s) {uncovered_ids} had no calibrant ring pixels.")
                lines.append("    Their shifts are fixed at 0 and could not be calibrated.")
                lines.append("    To calibrate these panels, use a geometry where rings")
                lines.append("    fall on those modules, or interpolate from neighbours.")
        else:
            lines.append("2. Panel Shift Correction")
            lines.append("─" * W)
            lines.append("   Not fitted — 'Optimize panels' was not enabled.")
        lines.append("")

        # ── 3. Distortion correction (non-flat detector surface) ──────────────
        dist_fitted = r.get('distortion_fitted', False)
        dist_note   = "fitted (p2, p3)" if dist_fitted else "not fitted — frozen at input values"
        lines.append("3. Distortion Correction (non-flat detector surface)")
        lines.append("─" * W)
        lines.append("   Corrects smooth surface warping on top of panel-corrected")
        lines.append("   geometry (curvature, indentation, arbitrary warping).")
        lines.append(f"   Status: {dist_note}")
        lines.append(f"  {'p0':6s}  {geom.get('p0', 0.0):>16.6g}      (2-fold azimuthal, cos 2η)")
        lines.append(f"  {'p1':6s}  {geom.get('p1', 0.0):>16.6g}      (4-fold azimuthal, cos 4η)")
        lines.append(f"  {'p2':6s}  {geom.get('p2', 0.0):>16.6g}      (isotropic radial)")
        lines.append(f"  {'p3':6s}  {geom.get('p3', 0.0):>16.6g}  deg (phase of p1 term)")
        lines.append(f"  {'p4':6s}  {geom.get('p4', 0.0):>16.6g}      (6th-order isotropic)")
        lines.append("")

        # ── 4. Fit quality ────────────────────────────────────────────────────
        lines.append("4. Fit Quality")
        lines.append("─" * W)
        lines.append("   Geometry residual on calibrant ring spots (not sample strain).")
        lines.append(f"  Mean |strain|  {r['mean_strain']*1e6:>10.1f}  ppm  (< 200 ppm: excellent)")
        lines.append(f"  Std  strain    {r['std_strain']*1e6:>10.1f}  ppm")
        lines.append(f"  Points used    {r['n_points']:>10d}")
        lines.append("")

        # ── 5. Per-ring strains ───────────────────────────────────────────────
        ring_strains = r.get('ring_strains', [])
        if ring_strains:
            lines.append("5. Per-Ring Strains")
            lines.append("─" * W)
            lines.append("   strain = (d_obs − d_ideal) / d_ideal = Δd/d")
            lines.append("   Systematic trend vs 2θ → wrong Lsd or BC.")
            lines.append("   Outlier ring → bad assignment or masked arc.")
            hdr = f"  {'2θ_ideal':>9}  {'2θ_obs':>9}  {'Δ2θ(deg)':>9}  {'strain(ppm)':>11}  {'n_pts':>6}"
            lines.append(hdr)
            lines.append("  " + "─" * (len(hdr) - 2))
            for rs in ring_strains:
                dt = rs['tth_observed'] - rs['tth_ideal']
                lines.append(
                    f"  {rs['tth_ideal']:9.4f}  {rs['tth_observed']:9.4f}"
                    f"  {dt:+8.4f}  {rs['strain']*1e6:+11.1f}  {rs['n_points']:6d}")

        return "\n".join(lines)

    # ── Export ────────────────────────────────────────────────────────────────

    def _on_export_params(self):
        if self._geom is None:
            QMessageBox.information(self, "Export", "Run calibration first.")
            return
        stem = (Path(self._image_path).stem if self._image_path else "geometry")
        default = str(Path(self._image_path).parent / f"{stem}_geom.toml") if self._image_path else ""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save params", default, "TOML files (*.toml);;All files (*)")
        if not path:
            return
        try:
            self._geom['panel_shifts'] = self._panel_shifts or []
            save_params(self._geom, path)
            log.info("Geometry saved: %s", path)
            n_ps = len(self._geom['panel_shifts'])
            detail = f" ({n_ps} panel shifts included)" if n_ps else ""
            QMessageBox.information(
                self, "Saved", f"Saved: {path}{detail}")
        except Exception as e:
            log.error("Save params failed: %s", e, exc_info=True)
            QMessageBox.critical(self, "Error", str(e))

    def _on_export_poni(self):
        if self._geom is None:
            QMessageBox.information(self, "Export .poni", "Run calibration first.")
            return
        stem = (Path(self._image_path).stem if self._image_path else "geometry")
        default = str(Path(self._image_path).parent / f"{stem}.poni") if self._image_path else f"{stem}.poni"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export .poni", default, "PONI files (*.poni);;All files (*)")
        if not path:
            return
        try:
            geom = dict(self._geom)
            geom.setdefault('nrows', int(self._nrows_ed.text()))
            write_poni(geom, path)
            log.info(".poni exported: %s", path)
            QMessageBox.information(self, "Exported", f"Saved: {path}")
        except Exception as e:
            log.error("Export .poni failed: %s", e, exc_info=True)
            QMessageBox.critical(self, "Error", str(e))

    def _on_export_midas_params(self):
        if self._geom is None:
            QMessageBox.information(self, "Export MIDAS params", "Run calibration first.")
            return
        stem = (Path(self._image_path).stem if self._image_path else "geometry")
        default = (str(Path(self._image_path).parent / f"{stem}_geom.txt")
                   if self._image_path else f"{stem}_geom.txt")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export MIDAS params", default,
            "MIDAS params (*.txt);;All files (*)")
        if not path:
            return
        geom = dict(self._geom)
        geom.setdefault('nrows', int(self._nrows_ed.text()))
        ps_path = None
        if geom.get('panel_shifts'):
            ps_path = str(Path(path).with_name(Path(path).stem + "_panel_shifts.txt"))
        try:
            from ..io import save_midas_params
            save_midas_params(geom, path, panel_shifts_path=ps_path)
            log.info("MIDAS params exported: %s", path)
            msg = f"Saved: {Path(path).name}"
            if ps_path:
                msg += f"\nPanel shifts: {Path(ps_path).name}"
            QMessageBox.information(self, "Exported", msg)
        except Exception as e:
            log.error("Export MIDAS params failed: %s", e, exc_info=True)
            QMessageBox.critical(self, "Error", str(e))

    def _on_save_lineout(self):
        if self._lineout is None:
            QMessageBox.information(self, "Save Lineout", "No lineout available yet.")
            return
        stem    = Path(self._image_path).stem if self._image_path else "lineout"
        default = (str(Path(self._image_path).parent / f"{stem}_lineout.xye")
                   if self._image_path else f"{stem}_lineout.xye")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Lineout", default,
            "XYE files (*.xye);;XY files (*.xy);;Text files (*.txt);;All files (*)")
        if not path:
            return
        tth, I, bg, I_sub, sigma, px_cnt = self._lineout
        try:
            wl = float(self._lambda_ed.text())
            e_kev = 12.3984193 / wl
            wl_hdr = f"  [wavelength={wl:.7f}A  energy={e_kev:.4f}keV]"
        except (ValueError, ZeroDivisionError):
            wl_hdr = ""
        cols   = [tth, I]
        header = "2theta_deg  I"
        if bg is not None:
            cols.append(bg)
            header += "  SNIP_bg"
        if I_sub is not None:
            cols.append(I_sub)
            header += "  I_sub"
        if sigma is not None:
            cols.append(sigma)
            header += "  sigma"
        if px_cnt is not None:
            cols.append(px_cnt)
            header += "  px_cnt"
        header += wl_hdr + "\n# sigma = sqrt(sum_counts)/n_active_pixels  (Poisson s.e.m.)"
        try:
            np.savetxt(path, np.column_stack(cols), header=header, fmt='%.6g')
            msg = f"Lineout saved: {path}"
            self._update_status_bar(msg)
            self._results_lbl.setText(msg)
        except Exception as e:
            log.error("Save lineout failed: %s", e, exc_info=True)
            QMessageBox.critical(self, "Error", str(e))

    def _on_save_cake(self):
        cake_img = (self._cake_canvas._image
                    if hasattr(self._cake_canvas, '_image') else None)
        if cake_img is None or self._cake_tth is None:
            QMessageBox.information(self, "Save Cake",
                                    "No cake image available — view the Cake page first.")
            return
        stem    = Path(self._image_path).stem if self._image_path else "cake"
        default = (str(Path(self._image_path).parent / f"{stem}_cake.tif")
                   if self._image_path else f"{stem}_cake.tif")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Cake", default,
            "TIFF files (*.tif *.tiff);;All files (*)")
        if not path:
            return
        try:
            import tifffile
            tifffile.imwrite(path, cake_img.astype(np.float32))
            axes_path = Path(path).with_suffix('.npz')
            np.savez(str(axes_path), tth=self._cake_tth, eta=self._cake_eta)
            msg = f"Cake saved: {Path(path).name}  (axes: {axes_path.name})"
            self._update_status_bar(msg)
            self._results_lbl.setText(msg)
        except Exception as e:
            log.error("Save cake failed: %s", e, exc_info=True)
            QMessageBox.critical(self, "Error", str(e))

    def _on_use_geometry(self):
        if self._geom is None:
            QMessageBox.information(
                self, "Use Geometry", "Run calibration first.")
            return
        self.geometry_ready.emit(self._geom)

    def apply_mpl_theme(self, colors):
        # ── Qt widget colours ──────────────────────────────────────────────────
        pos_color = colors.get('pos_label', '#111111')
        self._pos_label.setStyleSheet(
            f"color: {pos_color}; font-size: 13px; padding-left: 4px;")
        self._lo_pos_label.setStyleSheet(
            f"color: {pos_color}; font-size: 13px; padding-left: 8px;")
        # Results label: use the theme's foreground so it's readable on any bg
        self._results_lbl.setStyleSheet(
            f"color: {colors['fg']}; font-size: 10px;")

        # ── Matplotlib canvases ────────────────────────────────────────────────
        self._mpl_colors = colors   # stored so _refresh_lineout uses them
        self._raw_canvas.apply_mpl_theme(colors)
        self._cake_canvas.apply_mpl_theme(colors)

        # Lineout: redraw curves with new colours if data exists;
        # otherwise just restyle the empty axes.
        self.line_fig.set_facecolor(colors['fig'])
        if self._lineout is not None:
            self._refresh_lineout()   # clears + redraws with new colours
        else:
            self._style_lineout_ax()
            self.line_canvas.draw_idle()
        _refresh_navbar_icons(self._lo_toolbar)
