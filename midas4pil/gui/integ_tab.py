# Copyright (c) 2026, UChicago Argonne, LLC. All Rights Reserved.
# Author: Changyong Park, HPCAT, X-ray Science Division, Argonne National Laboratory.
# BSD Open Source License -- see LICENSE in the project root for the full license text.
"""
Integration tab — batch/manual mode data reduction (redesigned).

Layout
------
Left panel  (fixed 230 px, scrollable): file browser, mode controls
Centre      : CakeLineoutWidget  (TIFF/Cake toggle, crosshair, hist)

Lineout overlay
---------------
File-list checkboxes control which lineouts are overlaid on the lineout plot.
Cake image shows only the currently selected (clicked) file.
2θ x-axes of the cake image and the lineout plot are kept in sync.
"""

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                                QPushButton, QListWidget, QListWidgetItem,
                                QLabel, QFileDialog, QCheckBox,
                                QMessageBox, QRadioButton, QButtonGroup,
                                QGroupBox, QScrollArea, QSpinBox,
                                QDoubleSpinBox, QSplitter, QApplication)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeyEvent

from ..io import load_tiff, load_params, auto_mask, save_mask
from ..geometry import build_lut, lut_tth_range
from ..panels import build_lut_with_panels
from ..integrate import reduce_frame, precompute_bin_maps, rebin_lineout, snip_background, _HAS_NUMBA

from .display_widget import CakeLineoutWidget
from .worker import BatchWorker
from .crosshair import SyncBus


class _FileListWidget(QListWidget):
    """QListWidget with multi-select and drag-to-check gestures.

    Interactions
    ------------
    Click                 — select item and display it
    Ctrl+click            — add/remove item from selection
    Shift+click           — extend selection to clicked item
    Shift+scroll          — extend selection while scrolling (then Space to toggle)
    Space                 — check / uncheck all selected items
                            (checks all if any are unchecked; unchecks all otherwise)
    Left-drag on checkbox — sweep to set all dragged items to the same check state
                            as the first item after its toggle
    Right-click           — context menu: Check / Uncheck selected or all
    Ctrl+A                — select all items
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(self.SelectionMode.ExtendedSelection)
        self.setToolTip(
            "Click — select\n"
            "Shift+click — select range\n"
            "Ctrl+click — add to selection\n"
            "Ctrl+A — select all\n"
            "Space — check / uncheck selected\n"
            "Shift+scroll — extend selection while scrolling\n"
            "Left-drag on checkbox — sweep check / uncheck\n"
            "Right-click — context menu")
        self._drag_check: 'Qt.CheckState | None' = None
        self._drag_row = -1

    # ── Keyboard ──────────────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            items = self.selectedItems()
            if items:
                any_unch = any(
                    i.checkState() == Qt.CheckState.Unchecked for i in items)
                state = (Qt.CheckState.Checked if any_unch
                         else Qt.CheckState.Unchecked)
                for it in items:
                    it.setCheckState(state)
                return
        super().keyPressEvent(event)

    # ── Mouse ─────────────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        self._drag_check = None
        self._drag_row   = -1
        if event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.position().toPoint())
            if item is not None:
                rect = self.visualItemRect(item)
                in_checkbox = (event.position().x() - rect.left()) < 24
                super().mousePressEvent(event)   # Qt toggles checkbox
                if in_checkbox:
                    # Record the resulting check state to apply on drag
                    self._drag_check = item.checkState()
                    self._drag_row   = self.row(item)
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (self._drag_check is not None
                and event.buttons() & Qt.MouseButton.LeftButton):
            item = self.itemAt(event.position().toPoint())
            if item is not None:
                row = self.row(item)
                if row != self._drag_row:
                    item.setCheckState(self._drag_check)
                    self._drag_row = row
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_check = None
        self._drag_row   = -1
        super().mouseReleaseEvent(event)

    # ── Scroll ────────────────────────────────────────────────────────────────

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            # Extend selection in the scroll direction (like Shift+Arrow)
            delta = event.angleDelta().y()
            steps = max(1, abs(delta) // 120)
            key   = Qt.Key.Key_Up if delta > 0 else Qt.Key.Key_Down
            for _ in range(steps):
                fake = QKeyEvent(QKeyEvent.Type.KeyPress, key,
                                 Qt.KeyboardModifier.ShiftModifier)
                super().keyPressEvent(fake)
        else:
            super().wheelEvent(event)

    # ── Context menu ──────────────────────────────────────────────────────────

    def contextMenuEvent(self, event):
        from PySide6.QtWidgets import QMenu
        sel  = self.selectedItems()
        n    = len(sel)
        menu = QMenu(self)
        act_chk_sel   = menu.addAction(f"Check selected ({n})")
        act_unchk_sel = menu.addAction(f"Uncheck selected ({n})")
        menu.addSeparator()
        act_chk_all   = menu.addAction("Check all")
        act_unchk_all = menu.addAction("Uncheck all")
        act_chk_sel.setEnabled(n > 0)
        act_unchk_sel.setEnabled(n > 0)
        chosen = menu.exec(event.globalPos())
        if chosen == act_chk_sel:
            for it in sel:
                it.setCheckState(Qt.CheckState.Checked)
        elif chosen == act_unchk_sel:
            for it in sel:
                it.setCheckState(Qt.CheckState.Unchecked)
        elif chosen == act_chk_all:
            for i in range(self.count()):
                self.item(i).setCheckState(Qt.CheckState.Checked)
        elif chosen == act_unchk_all:
            for i in range(self.count()):
                self.item(i).setCheckState(Qt.CheckState.Unchecked)



class IntegrationTab(QWidget):
    """Integration tab with batch/manual mode and multi-lineout overlay display."""

    def __init__(self, sync_bus: SyncBus, parent=None):
        super().__init__(parent)
        self._sync_bus = sync_bus

        # ── State ──
        self.data_folder = None
        self.geom = None
        self.tth_lut = None
        self.eta_lut = None
        self.mask = None
        self.bin_maps = None
        self._mode = 'batch'
        self._integ_mode = 'varbin'
        self._known_files = set()
        self._pending_result = None
        self._worker = None
        # basename → (tth, I, bg, I_sub, sigma)  — data for items in the overlay
        self._lineout_store  = {}
        # basenames the user has explicitly checked (drives overlay + refresh check state)
        self._overlay_basenames: set = set()
        self._last_image_folder = None   # used as starting dir for Browse

        # Mask state
        self._use_mask      = False   # True when a mask is loaded and toggled on
        self._current_image = None    # last loaded/displayed image array
        self._current_base  = None    # stem of last displayed file
        self._mpl_colors    = None    # current theme colours (set by apply_mpl_theme)

        # ── Build UI ──
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Horizontal splitter: left file panel | right display area
        h_splitter = QSplitter(Qt.Orientation.Horizontal)
        h_splitter.setHandleWidth(4)
        h_splitter.setChildrenCollapsible(False)

        # Left: scrollable file panel (initially 230 px, user-resizable)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(160)
        left_scroll.setMaximumWidth(480)
        left_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setWidget(self._build_file_panel())
        h_splitter.addWidget(left_scroll)

        # Right: cake + lineout display (expands to fill available width)
        self.display = CakeLineoutWidget(sync_bus)
        self.display.add_to_header(self._btn_mask_toggle)
        self.display.eta_range_changed.connect(self._on_eta_drag_changed)
        self.display.tth_snip_range_changed.connect(self._on_tth_snip_changed)
        h_splitter.addWidget(self.display)

        h_splitter.setStretchFactor(0, 0)
        h_splitter.setStretchFactor(1, 1)
        h_splitter.setSizes([230, 900])
        main_layout.addWidget(h_splitter)

        # ── Timers ──
        self._display_timer = QTimer(self)
        self._display_timer.setInterval(200)   # 5 Hz display update cap
        self._display_timer.timeout.connect(self._flush_pending_display)

        self._watch_timer = QTimer(self)
        self._watch_timer.setInterval(1000)    # 1 Hz folder poll
        self._watch_timer.timeout.connect(self._poll_folder)

    # ── File Panel ──

    def _build_file_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Mode radio buttons
        mode_box = QGroupBox("Mode")
        mode_layout = QHBoxLayout(mode_box)
        mode_layout.setContentsMargins(4, 2, 4, 2)
        self._mode_group = QButtonGroup(self)
        self.radio_batch = QRadioButton("Batch")
        self.radio_manual = QRadioButton("Manual")
        self.radio_batch.setChecked(True)
        self.radio_batch.setToolTip(
            "Batch mode — click Start to process every .tif in the folder.\n"
            "Results are saved to lineouts/ and cakes/ subfolders.\n"
            "Enable 'Watch for new files' to auto-process incoming data.")
        self.radio_manual.setToolTip(
            "Manual mode — click any file in the list to process and display it.\n"
            "Useful for inspecting individual frames.  Results are not saved.")
        self._mode_group.addButton(self.radio_batch)
        self._mode_group.addButton(self.radio_manual)
        mode_layout.addWidget(self.radio_batch)
        mode_layout.addWidget(self.radio_manual)
        layout.addWidget(mode_box)

        self.radio_batch.toggled.connect(
            lambda checked: self.set_mode('batch') if checked else None)
        self.radio_manual.toggled.connect(
            lambda checked: self.set_mode('manual') if checked else None)

        # Integration mode radio buttons
        integ_box = QGroupBox("Integration")
        integ_layout = QHBoxLayout(integ_box)
        integ_layout.setContentsMargins(4, 2, 4, 2)
        self._integ_group = QButtonGroup(self)
        self.radio_varbin = QRadioButton("varbin")
        self.radio_unibin = QRadioButton("unibin")
        self.radio_varbin.setChecked(True)
        self.radio_varbin.setToolTip(
            "Variable-bin integration (default) — each bin spans the angular\n"
            "footprint of one detector pixel.  Physically correct; recommended\n"
            "for Rietveld refinement (GSAS-II, FullProf, TOPAS).")
        self.radio_unibin.setToolTip(
            "Uniform-bin integration — equal angular spacing.  Use only for\n"
            "PDF analysis (FFT requires uniform grid) or real-time monitoring\n"
            "where uniform spacing is required.")
        self._integ_group.addButton(self.radio_varbin)
        self._integ_group.addButton(self.radio_unibin)
        integ_layout.addWidget(self.radio_varbin)
        integ_layout.addWidget(self.radio_unibin)
        layout.addWidget(integ_box)

        self.radio_varbin.toggled.connect(
            lambda checked: self.set_integ_mode('varbin') if checked else None)
        self.radio_unibin.toggled.connect(
            lambda checked: self.set_integ_mode('unibin') if checked else None)

        # Rebin spinbox
        rebin_row = QHBoxLayout()
        rebin_lbl = QLabel("Rebin ×")
        self.spin_rebin = QSpinBox()
        self.spin_rebin.setMinimum(1)
        self.spin_rebin.setMaximum(32)
        self.spin_rebin.setValue(1)
        self.spin_rebin.setFixedWidth(52)
        self.spin_rebin.setToolTip(
            "Post-integration rebinning factor (N ≥ 1).\n"
            "N native bins are co-added with exact Poisson error propagation.\n"
            "SNIP background is re-computed on the coarsened lineout.\n"
            "Bin size is fixed at the instrument resolution floor — this\n"
            "controls display/export coarsening only.")
        self.spin_rebin.valueChanged.connect(self._recompute_current_lineout)
        rebin_row.addWidget(rebin_lbl)
        rebin_row.addWidget(self.spin_rebin)
        rebin_row.addStretch()
        layout.addLayout(rebin_row)

        # Eta bin size (varbin) with coupled 2th_min readout
        eta_bin_row = QHBoxLayout()
        eta_bin_lbl = QLabel("Eta bin (deg):")
        self.spin_eta_bin = QDoubleSpinBox()
        self.spin_eta_bin.setRange(0.1, 10.0)
        self.spin_eta_bin.setValue(1.0)
        self.spin_eta_bin.setDecimals(2)
        self.spin_eta_bin.setSingleStep(0.1)
        self.spin_eta_bin.setFixedWidth(66)
        self.spin_eta_bin.setToolTip(
            "Azimuthal bin width for the varbin cake image (default 1.00 deg).\n"
            "Coupled to the lower 2th cutoff via:\n"
            "   delta_eta [rad] = px / (Lsd * tan(2th_min))\n"
            "Finer bins raise 2th_min, excluding low-angle data\n"
            "from both the cake image and the lineout.")
        self.lbl_tth_min_info = QLabel("2th_min: -- deg")
        self.lbl_tth_min_info.setStyleSheet("font-size: 10px;")
        self.spin_eta_bin.valueChanged.connect(self._on_eta_bin_changed)
        eta_bin_row.addWidget(eta_bin_lbl)
        eta_bin_row.addWidget(self.spin_eta_bin)
        eta_bin_row.addWidget(self.lbl_tth_min_info)
        eta_bin_row.addStretch()
        layout.addLayout(eta_bin_row)

        # Warning label — shown only when eta bin is finer than default (< 1.0 deg)
        self.lbl_eta_bin_warn = QLabel("")
        self.lbl_eta_bin_warn.setWordWrap(True)
        self.lbl_eta_bin_warn.setStyleSheet("font-size: 9px; color: #d08000;")
        self.lbl_eta_bin_warn.setVisible(False)
        layout.addWidget(self.lbl_eta_bin_warn)

        # Eta sector range
        eta_row = QHBoxLayout()
        eta_lbl = QLabel("\u03b7 range:")
        self.spin_eta_min = QDoubleSpinBox()
        self.spin_eta_min.setRange(-180.0, 180.0)
        self.spin_eta_min.setValue(-180.0)
        self.spin_eta_min.setDecimals(1)
        self.spin_eta_min.setSingleStep(5.0)
        self.spin_eta_min.setFixedWidth(68)
        self.spin_eta_min.setToolTip(
            "Lower azimuthal limit for 1-D lineout integration.\n"
            "Drag the orange dashed line on the cake to adjust.\n"
            "Full range: \u2212180 \u2192 +180.")
        self.spin_eta_max = QDoubleSpinBox()
        self.spin_eta_max.setRange(-180.0, 180.0)
        self.spin_eta_max.setValue(180.0)
        self.spin_eta_max.setDecimals(1)
        self.spin_eta_max.setSingleStep(5.0)
        self.spin_eta_max.setFixedWidth(68)
        self.spin_eta_max.setToolTip(self.spin_eta_min.toolTip())
        self.spin_eta_min.valueChanged.connect(self._on_eta_changed)
        self.spin_eta_max.valueChanged.connect(self._on_eta_changed)
        eta_row.addWidget(eta_lbl)
        eta_row.addWidget(self.spin_eta_min)
        eta_row.addWidget(QLabel("\u2013"))
        eta_row.addWidget(self.spin_eta_max)
        layout.addLayout(eta_row)

        # Folder browse
        self.btn_browse = QPushButton("Browse folder...")
        self.btn_browse.clicked.connect(self._on_browse)
        layout.addWidget(self.btn_browse)

        self.lbl_folder = QLabel("No folder selected")
        self.lbl_folder.setWordWrap(True)
        self.lbl_folder.setStyleSheet("font-size: 10px;")
        layout.addWidget(self.lbl_folder)

        # File list (items are checkable — check to include in lineout overlay)
        self.file_list = _FileListWidget()
        self.file_list.setToolTip(
            "File list — click to display; check to overlay in lineout plot.\n"
            "Space        — toggle check state of selected files\n"
            "Shift+scroll — extend selection up / down\n"
            "Drag         — drag over items to check/uncheck them\n"
            "Right-click  — context menu: check/uncheck selected or all")
        self.file_list.currentRowChanged.connect(self._on_file_selected)
        self.file_list.itemChanged.connect(self._on_item_check_changed)
        layout.addWidget(self.file_list, stretch=1)

        # Navigation
        nav_row = QHBoxLayout()
        btn_prev = QPushButton("\u25c4 Prev")
        btn_prev.clicked.connect(self._on_prev)
        btn_next = QPushButton("Next \u25ba")
        btn_next.clicked.connect(self._on_next)
        nav_row.addWidget(btn_prev)
        nav_row.addWidget(btn_next)
        layout.addLayout(nav_row)

        # Watch checkbox
        self.chk_watch = QCheckBox("Watch for new files")
        self.chk_watch.setToolTip(
            "After batch processing finishes, keep watching the folder\n"
            "and automatically process new .tif files as they appear.")
        layout.addWidget(self.chk_watch)

        # Start / Stop
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_start.setToolTip(
            "Start batch processing — reduce every .tif in the folder\n"
            "and save results to lineouts/ and cakes/ subfolders.")
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setToolTip("Stop the running batch job.")
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        layout.addLayout(btn_row)

        # Export I_sub
        self.btn_export_isub = QPushButton("Export I_sub…")
        self.btn_export_isub.setToolTip(
            "Re-integrate with current SNIP 2\u03b8 and \u03b7 ranges.\n"
            "Saves [2\u03b8, I, SNIP_bg, I_sub, \u03c3, px_cnt] to lineouts_sub/.\n"
            "Choose scope: current file, checked overlay files, or all files.")
        self.btn_export_isub.clicked.connect(self._on_export_i_sub)
        self.btn_export_isub.setEnabled(False)
        layout.addWidget(self.btn_export_isub)

        # Mask toggle (added to display header) + editor (in left panel)
        self._btn_mask_toggle = QPushButton("Mask \u25cb")
        self._btn_mask_toggle.setCheckable(True)
        self._btn_mask_toggle.setFixedWidth(64)
        self._btn_mask_toggle.toggled.connect(self._on_toggle_mask)
        mask_row = QHBoxLayout()
        self._btn_mask_editor = QPushButton("Mask\u2026")
        self._btn_mask_editor.clicked.connect(self._on_open_mask_editor)
        mask_row.addWidget(self._btn_mask_editor)
        layout.addLayout(mask_row)

        # Progress label
        self.lbl_progress = QLabel("")
        self.lbl_progress.setWordWrap(True)
        self.lbl_progress.setStyleSheet("font-size: 10px;")
        layout.addWidget(self.lbl_progress)

        return panel

    # ── Public API ──

    def load_mask(self, mask):
        """Load a mask array (called from main window when geometry is applied).

        Parameters
        ----------
        mask : bool array or None
        """
        self.mask = mask.copy() if mask is not None else None
        self._use_mask = self.mask is not None
        self._btn_mask_toggle.setChecked(self._use_mask)
        self._btn_mask_toggle.setText(
            "Mask \u25cf" if self._use_mask else "Mask \u25cb")
        self._recompute_bin_maps()
        self.display.show_mask_overlay(
            self.mask if self._use_mask else None, self._use_mask)
        self._recompute_current_lineout()

    def set_mode(self, mode):
        self._mode = mode
        self.btn_start.setEnabled(mode == 'batch')
        self.btn_stop.setEnabled(False)
        self.chk_watch.setEnabled(mode == 'batch')
        if mode == 'batch' and not self.radio_batch.isChecked():
            self.radio_batch.setChecked(True)
        elif mode == 'manual' and not self.radio_manual.isChecked():
            self.radio_manual.setChecked(True)

    def set_integ_mode(self, integ_mode):
        self._integ_mode = integ_mode
        if self.geom is not None:
            self.geom['mode'] = integ_mode
            self._recompute_bin_maps()
        if integ_mode == 'varbin' and not self.radio_varbin.isChecked():
            self.radio_varbin.setChecked(True)
        elif integ_mode == 'unibin' and not self.radio_unibin.isChecked():
            self.radio_unibin.setChecked(True)

    def load_geometry(self, geom):
        """Load geometry dict (from calibration tab or file).

        Panel shifts are read from geom['panel_shifts'] (list of dicts,
        may be empty).  The panel_map is derived on the fly from the
        detector shape when panel shifts are present.
        """
        # If nrows/ncols are absent (MIDAS params file without NrPixels lines),
        # inherit from the current geometry so the LUT build doesn't fail.
        if ('nrows' not in geom or 'ncols' not in geom) and self.geom is not None:
            geom = dict(geom)
            geom.setdefault('nrows', self.geom.get('nrows'))
            geom.setdefault('ncols', self.geom.get('ncols'))

        lut_keys = ['nrows', 'ncols', 'bc_y', 'bc_z', 'lsd', 'px',
                    'tx_deg', 'ty_deg', 'tz_deg',
                    'p0', 'p1', 'p2', 'p3', 'p4', 'rho_d']
        # Build LUT before committing state — keeps self.geom/tth_lut consistent on error.
        lut_kw = {k: geom[k] for k in lut_keys}

        panel_shifts = geom.get('panel_shifts', [])
        if panel_shifts:
            from .detectors import make_panel_map_from_shape
            panel_map = make_panel_map_from_shape(
                int(geom['nrows']), int(geom['ncols']))
            if panel_map is not None:
                tth_lut, eta_lut = build_lut_with_panels(
                    **lut_kw, panel_map=panel_map, panel_shifts=panel_shifts)
                log.info("LUT built with %d panel corrections", len(panel_shifts))
            else:
                log.warning("Panel shifts present but no panel preset matches "
                            "%dx%d — building LUT without corrections",
                            geom['nrows'], geom['ncols'])
                tth_lut, eta_lut = build_lut(**lut_kw)
        else:
            tth_lut, eta_lut = build_lut(**lut_kw)

        # Commit state only after successful LUT build.
        self.geom = geom
        self.geom['mode'] = self._integ_mode
        self.tth_lut, self.eta_lut = tth_lut, eta_lut

        # Derive integration limits from actual detector coverage.
        # Reset eta bin spinbox to default on new geometry load.
        self.spin_eta_bin.blockSignals(True)
        self.spin_eta_bin.setValue(1.0)
        self.spin_eta_bin.blockSignals(False)
        tth_min, tth_max = lut_tth_range(self.tth_lut,
                                         px=self.geom.get('px'),
                                         lsd=self.geom.get('lsd'),
                                         max_eta_bin_deg=1.0)
        self.geom['tth_min'] = tth_min
        self.geom['tth_max'] = tth_max
        self.geom.pop('eta_bin_size', None)   # clear any stale value
        self._update_eta_bin_labels(tth_min)

        # Pass LUTs to display widget for TIFF hover lookup
        self.display.set_luts(self.tth_lut, self.eta_lut)
        # Reset SNIP 2θ range to full detector coverage
        self.display.set_tth_snip_range(tth_min, tth_max)
        # Reset eta sector to full range
        self.spin_eta_min.blockSignals(True)
        self.spin_eta_max.blockSignals(True)
        self.spin_eta_min.setValue(-180.0)
        self.spin_eta_max.setValue(180.0)
        self.spin_eta_min.blockSignals(False)
        self.spin_eta_max.blockSignals(False)
        self.display.set_eta_lines(-180.0, 180.0)

        self.mask = None
        self._recompute_bin_maps()

        # Discard any cached lineouts — they were computed with the previous
        # geometry and are now stale.  Uncheck all file-list items and clear
        # the overlay so the user sees a clean slate and knows to rerun.
        had_results = bool(self._lineout_store or self._overlay_basenames)
        self._lineout_store.clear()
        self._overlay_basenames.clear()
        self.display.clear_lineouts()
        if self.file_list.count():
            self.file_list.blockSignals(True)
            for row in range(self.file_list.count()):
                self.file_list.item(row).setCheckState(Qt.CheckState.Unchecked)
            self.file_list.blockSignals(False)

        msg = ("Geometry updated — rerun batch to recalculate lineouts"
               if had_results else "Geometry loaded")
        self.lbl_progress.setText(msg)

    def load_geometry_file(self, path):
        geom = load_params(path)
        self.load_geometry(geom)

    # ── Internal ──

    def _recompute_bin_maps(self):
        self.bin_maps = None
        if not _HAS_NUMBA or self.geom is None:
            return
        if self.tth_lut is None or self.eta_lut is None:
            return
        # There is always a mask — either the real mask (when enabled) or a
        # null mask (all-zeros, no bad pixels).  bin_maps are valid for both,
        # so the numba path is always active once geometry is loaded.
        null_mask = np.zeros(self.tth_lut.shape, dtype=bool)
        mask = self.mask if (self.mask is not None and self._use_mask) else null_mask
        self.bin_maps = precompute_bin_maps(
            mask, self.tth_lut, self.eta_lut, self.geom)

    def _ensure_mask(self, image):
        if self.mask is not None:
            return
        from .detectors import make_panel_map_from_shape
        panel_map = make_panel_map_from_shape(*image.shape)
        if panel_map is not None:
            self.mask = auto_mask(image, panel_map=panel_map)
            self.mask = self.mask | (panel_map == 0)
        else:
            self.mask = auto_mask(image)
        self._recompute_bin_maps()

    def _refresh_file_list(self):
        # Block signals during rebuild to avoid spurious itemChanged
        self.file_list.blockSignals(True)
        self.file_list.clear()
        if self.data_folder is None:
            self.file_list.blockSignals(False)
            return
        tif_files = sorted(self.data_folder.glob('*.tif'))
        cakes_dir = self.data_folder / 'cakes'
        tif_files = [f for f in tif_files if f.parent != cakes_dir]
        self._known_files = set()
        for f in tif_files:
            base = f.stem
            self._known_files.add(f.name)
            has_result = (
                (self.data_folder / 'lineouts'        / f'{base}.xye').is_file() or
                (self.data_folder / 'lineouts_nomask' / f'{base}.xye').is_file())
            item = QListWidgetItem(f.name + ('  \u2713' if has_result else ''))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            # Checked = user explicitly added this frame to the overlay
            item.setCheckState(
                Qt.CheckState.Checked if base in self._overlay_basenames
                else Qt.CheckState.Unchecked)
            self.file_list.addItem(item)
        # Qt silently sets row 0 as current when the first item is added to an
        # empty list, even while signals are blocked.  Reset to -1 so that the
        # user's first click on row 0 actually triggers currentRowChanged(0).
        self.file_list.setCurrentRow(-1)
        self.file_list.blockSignals(False)

    # ── Callbacks ──

    def _on_browse(self):
        start_dir = str(
            self._last_image_folder or self.data_folder or Path.home())
        folder = QFileDialog.getExistingDirectory(
            self, "Select data folder", start_dir)
        if folder:
            self.data_folder = Path(folder)
            self.lbl_folder.setText(str(self.data_folder))
            self._lineout_store.clear()
            self.display.clear_lineouts()
            self._refresh_file_list()

    def _on_file_selected(self, row):
        if row < 0 or self.data_folder is None:
            return
        item = self.file_list.item(row)
        if item is None:
            return
        filename = item.text().replace('  \u2713', '')
        base = Path(filename).stem

        out_suffix   = '' if self._use_mask else '_nomask'
        lineout_path = self.data_folder / f'lineouts{out_suffix}' / f'{base}.xye'
        if not lineout_path.is_file():
            # Fallback to the other suffix in case mask state changed since batch
            alt = '_nomask' if self._use_mask else ''
            alt_path = self.data_folder / f'lineouts{alt}' / f'{base}.xye'
            if alt_path.is_file():
                lineout_path = alt_path
        cake_path    = self.data_folder / 'cakes'    / f'{base}.tif'

        # Always try to populate _current_image from the raw TIFF
        tif_path = self.data_folder / filename
        if tif_path.is_file():
            try:
                self._current_image = load_tiff(tif_path).astype(np.float64)
                self._current_base = base
            except Exception as e:
                log.warning("TIFF load failed for %s: %s", tif_path.name, e)

        if lineout_path.is_file() or cake_path.is_file():
            self.display.show_pair(base, self.data_folder)
            # Recompute lineout with the current mask so the displayed bg
            # always reflects the active mask — not stale on-disk values.
            if self.geom is not None and self.tth_lut is not None:
                self._recompute_current_lineout()
        elif self._mode == 'manual' and self.geom is not None:
            self._process_single(self.data_folder / filename)

    def _on_prev(self):
        row = self.file_list.currentRow()
        if row > 0:
            self.file_list.setCurrentRow(row - 1)

    def _on_next(self):
        row = self.file_list.currentRow()
        if row < self.file_list.count() - 1:
            self.file_list.setCurrentRow(row + 1)

    def _on_item_check_changed(self, item):
        """Add or remove this file's lineout from the overlay when checkbox changes."""
        fname = item.text().replace('  \u2713', '')
        base  = Path(fname).stem
        if item.checkState() == Qt.CheckState.Checked:
            self._overlay_basenames.add(base)
            # Try in-memory store first, then load from disk
            data = self._lineout_store.get(base)
            if data is None and self.data_folder is not None:
                out_sfx = '' if self._use_mask else '_nomask'
                xye = self.data_folder / f'lineouts{out_sfx}' / f'{base}.xye'
                if not xye.is_file():
                    alt = '_nomask' if self._use_mask else ''
                    alt_xye = self.data_folder / f'lineouts{alt}' / f'{base}.xye'
                    if alt_xye.is_file():
                        xye = alt_xye
                if xye.is_file():
                    try:
                        arr = np.loadtxt(str(xye))
                        if arr.ndim == 2 and arr.shape[1] >= 6:
                            # New I_sub format: 2θ  I  SNIP_bg  I_sub  σ  px_cnt
                            data = (arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3],
                                    arr[:, 4], arr[:, 5].astype(int))
                        elif arr.ndim == 2 and arr.shape[1] == 5:
                            # Legacy I_sub (no px_cnt): 2θ  I  SNIP_bg  I_sub  σ
                            data = (arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3],
                                    arr[:, 4], None)
                        elif arr.ndim == 2 and arr.shape[1] == 4:
                            # New I-export: 2θ  I  σ  px_cnt
                            data = (arr[:, 0], arr[:, 1], None, None,
                                    arr[:, 2], arr[:, 3].astype(int))
                        elif arr.ndim == 2 and arr.shape[1] >= 3:
                            # Legacy I-export: 2θ  I  σ
                            data = (arr[:, 0], arr[:, 1], None, None,
                                    arr[:, 2], None)
                    except Exception as e:
                        log.debug("XY file parse failed for %s: %s", xye.name, e)
                        data = None
            if data is not None:
                self._lineout_store[base] = data
                self.display.add_lineout(base, *data)
        else:
            self._overlay_basenames.discard(base)
            self._lineout_store.pop(base, None)
            self.display.remove_lineout(base)

    def _on_start(self):
        if self.geom is None:
            QMessageBox.warning(self, "No geometry", "Load a geometry file first.")
            return
        if self.tth_lut is None or self.eta_lut is None:
            QMessageBox.warning(self, "No geometry", "Geometry LUT not built.")
            return
        if self.data_folder is None:
            QMessageBox.warning(self, "No folder", "Select a data folder first.")
            return

        tif_files = sorted(self.data_folder.glob('*.tif'))
        cakes_dir = self.data_folder / 'cakes'
        tif_files = [f for f in tif_files if f.parent != cakes_dir]
        if not tif_files:
            QMessageBox.warning(self, "No files", "No .tif files found.")
            return

        try:
            first_image = load_tiff(tif_files[0]).astype(np.float64)
            self._ensure_mask(first_image)
        except Exception as e:
            log.error("Failed to load first image %s: %s", tif_files[0].name, e, exc_info=True)
            QMessageBox.warning(self, "Error", f"Failed to load first image:\n{e}")
            return

        eff_mask = (self.mask if (self.mask is not None and self._use_mask)
                    else np.zeros(first_image.shape, dtype=bool))
        out_suffix = '' if self._use_mask else '_nomask'

        self._worker = BatchWorker()
        self._worker.setup(
            self.data_folder, eff_mask,
            self.tth_lut, self.eta_lut, self.geom,
            bin_maps=self.bin_maps,
            out_suffix=out_suffix,
            eta_min=self.spin_eta_min.value(),
            eta_max=self.spin_eta_max.value())
        self._worker.frame_processed.connect(self._on_frame_processed)
        self._worker.progress.connect(self._on_progress)
        self._worker.batch_finished.connect(self._on_batch_finished)
        self._worker.error.connect(self._on_error)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._display_timer.start()
        self._worker.start()

        if not self._use_mask:
            self.lbl_progress.setText(
                "Mask off — saving to lineouts_nomask/ (no cake written)")

        if self.chk_watch.isChecked():
            self._watch_timer.start()

    def _on_stop(self):
        if self._worker is not None:
            self._worker.request_stop()
        self._watch_timer.stop()
        self.btn_stop.setEnabled(False)

    def _on_frame_processed(self, result, basename):
        self._pending_result = (result, basename)
        # Track last processed image for mask editor
        if 'image' in result:
            self._current_image = result['image']
            self._current_base  = basename
        if 'tth' in result and 'I' in result:
            entry = (result['tth'], result['I'],
                     result.get('bg'), result.get('I_sub'), result.get('sigma'),
                     result.get('px_cnt'))
            self._lineout_store[basename] = entry
            # Do NOT auto-check: with hundreds of files the overlay overflows.
            # Users select which frames to plot by checking items manually.

    def _on_progress(self, current, total):
        self.lbl_progress.setText(f"{current} / {total}")

    def _on_batch_finished(self, count):
        # Guarded: an unhandled exception here propagates into Qt's event loop
        # (potentially during a modal dialog's nested loop on Windows), which
        # can collapse the main window.
        try:
            self._display_timer.stop()
            self._flush_pending_display()
            self._refresh_file_list()
        except Exception as e:
            log.error("_on_batch_finished display update failed: %s", e, exc_info=True)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_progress.setText(f"Done: {count} frames")
        if self.chk_watch.isChecked():
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)

    def _on_error(self, msg):
        try:
            self.lbl_progress.setText(f"Error: {msg}")
        except Exception as e:
            log.error("_on_error slot failed: %s", e, exc_info=True)

    def _flush_pending_display(self):
        if self._pending_result is None:
            return
        result, basename = self._pending_result
        self._pending_result = None
        self.display.show_results(result, title=basename)

    def _poll_folder(self):
        if self.data_folder is None:
            return
        current_files = {f.name for f in self.data_folder.glob('*.tif')
                         if f.parent != self.data_folder / 'cakes'}
        new_files = current_files - self._known_files
        if not new_files:
            return
        self._known_files = current_files
        new_paths = [self.data_folder / f for f in sorted(new_files)]

        if self._worker is None or not self._worker.isRunning():
            if self.mask is not None and self._use_mask:
                eff_mask = self.mask
            elif self.tth_lut is not None:
                eff_mask = np.zeros(self.tth_lut.shape, dtype=bool)
            else:
                return   # no geometry yet — cannot process
            out_suffix = '' if self._use_mask else '_nomask'
            self._worker = BatchWorker()
            self._worker.setup(
                self.data_folder, eff_mask,
                self.tth_lut, self.eta_lut, self.geom,
                bin_maps=self.bin_maps,
                out_suffix=out_suffix,
                eta_min=self.spin_eta_min.value(),
                eta_max=self.spin_eta_max.value())
            self._worker.set_file_list(new_paths)
            self._worker.frame_processed.connect(self._on_frame_processed)
            self._worker.progress.connect(self._on_progress)
            self._worker.batch_finished.connect(self._on_batch_finished)
            self._worker.error.connect(self._on_error)
            self._display_timer.start()
            self._worker.start()

    def _process_single(self, tif_path, save_output=True):
        """Process a single TIFF and optionally save lineout + cake.

        Parameters
        ----------
        tif_path   : Path to .tif file
        save_output: if True (default), write lineout and cake to disk;
                     if False, only update display (no disk write)
        """
        if self.geom is None or self.tth_lut is None or self.eta_lut is None:
            self.lbl_progress.setText("No geometry loaded")
            return
        try:
            image = load_tiff(tif_path).astype(np.float64)
            self._current_image = image
            self._current_base  = tif_path.stem
            self._ensure_mask(image)

            eff_mask = (self.mask if (self.mask is not None and self._use_mask)
                        else np.zeros(image.shape, dtype=bool))

            result = reduce_frame(
                image, eff_mask,
                self.tth_lut, self.eta_lut, self.geom,
                bin_maps=self.bin_maps,
                eta_min=self.spin_eta_min.value(),
                eta_max=self.spin_eta_max.value())
            self._apply_rebin(result)
            self._apply_snip_range(result)
            result['image'] = image   # store for TIFF toggle
            self.btn_export_isub.setEnabled(True)

            base = tif_path.stem
            self.display.show_results(result, title=base)
            # Update lineout overlay store for manual-mode results
            if 'tth' in result and 'I' in result:
                entry = (result['tth'], result['I'],
                         result.get('bg'), result.get('I_sub'), result.get('sigma'),
                         result.get('px_cnt'))
                self._lineout_store[base] = entry

            if not save_output:
                return

            out_suffix  = '' if self._use_mask else '_nomask'
            lineout_dir = self.data_folder / f'lineouts{out_suffix}'
            cake_dir    = self.data_folder / 'cakes'
            lineout_dir.mkdir(exist_ok=True)
            save_cake = (out_suffix == '')
            if save_cake:
                cake_dir.mkdir(exist_ok=True)

            sigma  = result.get('sigma',  np.full_like(result['I'], np.nan))
            px_cnt = result.get('px_cnt', np.full(len(result['I']), -1, dtype=int))
            wl = self.geom.get('wavelength', float('nan'))
            e_kev = 12.3984193 / wl if wl else float('nan')
            np.savetxt(
                str(lineout_dir / f'{base}.xye'),
                np.column_stack([result['tth'], result['I'], sigma, px_cnt]),
                header=(f'col1=2theta_deg  col2=I  col3=sigma_I  col4=px_cnt'
                        f'  [wavelength={wl:.7f}A  energy={e_kev:.4f}keV'
                        f'  eta={self.spin_eta_min.value():.1f}-{self.spin_eta_max.value():.1f}deg]'),
                fmt='%.6f')

            if save_cake:
                import tifffile
                tifffile.imwrite(
                    str(cake_dir / f'{base}.tif'),
                    result['cake_img'].T.astype(np.float32))
                # Save 2theta and eta axes so the cake can be displayed with correct axes
                if 'tth_cake' in result and 'eta_cake' in result:
                    np.savez(str(cake_dir / f'{base}_axes.npz'),
                             tth=result['tth_cake'], eta=result['eta_cake'])

            self._refresh_file_list()
            saved_msg = f"lineouts{out_suffix}/{base}.xye"
            if save_cake:
                saved_msg += f"  |  cakes/{base}.tif"
            self.lbl_progress.setText(f"Saved: {saved_msg}")

        except Exception as e:
            log.error("Single-file process/save failed for %s: %s",
                      self._current_base or '?', e, exc_info=True)
            self.lbl_progress.setText(f"Error: {e}")


    # ── Mask actions ────────────────────────────────────────────────────────

    def _on_toggle_mask(self, checked):
        self._use_mask = checked
        self._btn_mask_toggle.setText(
            "Mask \u25cf" if checked else "Mask \u25cb")
        show = checked and self.mask is not None
        self.display.show_mask_overlay(self.mask if show else None, show)
        self._recompute_bin_maps()   # rebuild for real mask or null mask

        if not checked and self._current_image is not None:
            # Mask turned OFF — ask whether to save nomask results
            has_data = (self.data_folder is not None and
                        self._current_base is not None)
            if has_data:
                msg = QMessageBox(self)
                msg.setWindowTitle("Mask turned off")
                msg.setText(
                    "The mask has been turned off. Would you like to "
                    "recalculate the lineout without the mask?\n\n"
                    "New lineouts will be saved in the lineouts_nomask/ folder.")
                btn_current = msg.addButton(
                    "Current file only", QMessageBox.ButtonRole.AcceptRole)
                btn_all = msg.addButton(
                    "All files in folder", QMessageBox.ButtonRole.AcceptRole)
                msg.addButton("Display only", QMessageBox.ButtonRole.RejectRole)
                msg.exec()
                clicked = msg.clickedButton()
                if clicked == btn_current:
                    self._reprocess_and_save_current()
                elif clicked == btn_all:
                    self._on_start()

        self._recompute_current_lineout()

    def _on_open_mask_editor(self):
        if self._current_image is None:
            QMessageBox.information(
                self, "Mask Editor",
                "Select a file from the list first to provide the image.")
            return
        old_mask = self.mask.copy() if self.mask is not None else None
        from .mask_editor import MaskEditorDialog
        from .detectors import make_panel_map_from_shape
        panel_map = make_panel_map_from_shape(*self._current_image.shape)
        dlg = MaskEditorDialog(self._current_image, mask=self.mask,
                               panel_map=panel_map,
                               colors=self._mpl_colors, parent=self)
        if dlg.exec() == 1:   # QDialog.DialogCode.Accepted == 1
            new_mask = dlg.result_mask()
            if new_mask is not None:
                mask_changed = (old_mask is None or
                                not np.array_equal(new_mask, old_mask))
                self.mask = new_mask
                self._use_mask = True
                self._btn_mask_toggle.setChecked(True)
                self._btn_mask_toggle.setText("Mask \u25cf")
                self._recompute_bin_maps()
                self.display.show_mask_overlay(self.mask, True)
                if mask_changed:
                    msg = QMessageBox(self)
                    msg.setWindowTitle("Mask changed")
                    msg.setText(
                        "The mask has changed. How would you like to "
                        "recalculate the lineout?")
                    btn_current = msg.addButton(
                        "Current file only", QMessageBox.ButtonRole.AcceptRole)
                    btn_all = msg.addButton(
                        "All files in folder", QMessageBox.ButtonRole.AcceptRole)
                    btn_display = msg.addButton(
                        "Display only", QMessageBox.ButtonRole.RejectRole)
                    msg.exec()
                    clicked = msg.clickedButton()
                    if clicked == btn_current:
                        self._reprocess_and_save_current()
                    elif clicked == btn_all:
                        self._reprocess_and_save_current()
                        self._on_start()
                    # else: display only — fall through to recompute below
                self._recompute_current_lineout()

    def _on_eta_bin_changed(self, value):
        """Eta bin spinbox changed — recompute tth_min and bin maps."""
        if self.tth_lut is None or self.geom is None:
            self._update_eta_bin_labels(None, value)
            return
        tth_min, _ = lut_tth_range(self.tth_lut,
                                   px=self.geom.get('px'),
                                   lsd=self.geom.get('lsd'),
                                   max_eta_bin_deg=value)
        self.geom['tth_min'] = tth_min
        if value < 1.0:
            # Store explicit eta_bin_size so reduce_frame uses it directly.
            self.geom['eta_bin_size'] = value
        else:
            self.geom.pop('eta_bin_size', None)
        self.display.set_tth_snip_range(tth_min, self.geom.get('tth_max', tth_min + 30.0))
        self._update_eta_bin_labels(tth_min, value)
        self._recompute_bin_maps()
        self._recompute_current_lineout()

    def _update_eta_bin_labels(self, tth_min, eta_bin=None):
        """Refresh the 2th_min readout and warning label."""
        if tth_min is None:
            self.lbl_tth_min_info.setText("2th_min: -- deg")
            self.lbl_eta_bin_warn.setVisible(False)
            return
        self.lbl_tth_min_info.setText(f"2th_min: {tth_min:.2f} deg")
        if eta_bin is None:
            eta_bin = self.spin_eta_bin.value()
        if eta_bin < 1.0:
            self.lbl_eta_bin_warn.setText(
                f"Finer eta bin: data below {tth_min:.2f} deg excluded "
                f"from cake and lineout.")
            self.lbl_eta_bin_warn.setVisible(True)
        else:
            self.lbl_eta_bin_warn.setVisible(False)

    def _on_eta_changed(self):
        lo = self.spin_eta_min.value()
        hi = self.spin_eta_max.value()
        self.display.set_eta_lines(lo, hi)
        self._recompute_current_lineout()
        self._recompute_all_overlays()

    def _on_eta_drag_changed(self, lo, hi):
        self.spin_eta_min.blockSignals(True)
        self.spin_eta_max.blockSignals(True)
        self.spin_eta_min.setValue(lo)
        self.spin_eta_max.setValue(hi)
        self.spin_eta_min.blockSignals(False)
        self.spin_eta_max.blockSignals(False)
        self._recompute_current_lineout()
        self._recompute_all_overlays()

    def _on_export_i_sub(self):
        """Scope-dialog re-integration: save 6-col [2θ,I,SNIP_bg,I_sub,σ,px_cnt] files."""
        if self.geom is None or self.data_folder is None:
            return

        tth_lo = self.display.tth_snip_lo
        tth_hi = self.display.tth_snip_hi
        eta_lo = self.spin_eta_min.value()
        eta_hi = self.spin_eta_max.value()

        # Scope dialog
        msg = QMessageBox(self)
        msg.setWindowTitle("Export I_sub — choose scope")
        msg.setText("Re-integrate and save full 6-column I_sub files.\nChoose which files to process:")
        btn_cur     = msg.addButton("Current file",          QMessageBox.ButtonRole.AcceptRole)
        btn_checked = msg.addButton("Checked overlay files", QMessageBox.ButtonRole.AcceptRole)
        btn_all     = msg.addButton("All files in folder",   QMessageBox.ButtonRole.AcceptRole)
        msg.addButton(QMessageBox.StandardButton.Cancel)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked is None or clicked not in (btn_cur, btn_checked, btn_all):
            return

        # Build file list
        if clicked is btn_cur:
            if self._current_image is None or self._current_base is None:
                self.lbl_progress.setText("No current file.")
                return
            file_list = [(self._current_base, self._current_image)]
        elif clicked is btn_checked:
            file_list = []
            for base in sorted(self._overlay_basenames):
                tif = self.data_folder / f'{base}.tif'
                if not tif.is_file():
                    continue
                try:
                    import tifffile
                    file_list.append((base, load_tiff(str(tif)).astype(np.float64)))
                except Exception as e:
                    log.warning("Could not load %s: %s", tif.name, e)
        else:  # All files
            file_list = []
            import tifffile
            cakes_dir = self.data_folder / 'cakes'
            for tif in sorted(self.data_folder.glob('*.tif')):
                if tif.parent == cakes_dir:
                    continue
                try:
                    file_list.append((tif.stem, load_tiff(str(tif)).astype(np.float64)))
                except Exception as e:
                    log.warning("Could not load %s: %s", tif.name, e)

        if not file_list:
            self.lbl_progress.setText("No files to export.")
            return

        out_suffix = '' if self._use_mask else '_nomask'
        sub_dir = self.data_folder / f'lineouts_sub{out_suffix}'
        sub_dir.mkdir(exist_ok=True)

        wl    = self.geom.get('wavelength', float('nan'))
        e_kev = 12.3984193 / wl if wl else float('nan')
        n     = len(file_list)

        for k, (base, image) in enumerate(file_list):
            self.lbl_progress.setText(f"Exporting I_sub {k+1}/{n}: {base}…")
            QApplication.processEvents()
            try:
                eff_mask = (self.mask if (self.mask is not None and self._use_mask)
                            else np.zeros(image.shape, dtype=bool))
                sub_geom = dict(self.geom)
                sub_geom['tth_min'] = tth_lo
                sub_geom['tth_max'] = tth_hi
                result = reduce_frame(image, eff_mask,
                                      self.tth_lut, self.eta_lut, sub_geom,
                                      cake_out=False, eta_min=eta_lo, eta_max=eta_hi)
                tth    = result['tth']
                I      = result['I']
                bg     = result.get('bg',     np.full_like(I, np.nan))
                I_sub  = result.get('I_sub',  np.full_like(I, np.nan))
                sigma  = result.get('sigma',  np.full_like(I, np.nan))
                px_cnt = result.get('px_cnt', np.full(len(I), -1, dtype=int))
                hdr = (f'col1=2theta_deg  col2=I  col3=SNIP_bg  col4=I_sub'
                       f'  col5=sigma_I  col6=px_cnt'
                       f'  [wavelength={wl:.7f}A  energy={e_kev:.4f}keV'
                       f'  2th={tth_lo:.3f}-{tth_hi:.3f}deg'
                       f'  eta={eta_lo:.1f}-{eta_hi:.1f}deg]')
                np.savetxt(
                    str(sub_dir / f'{base}.xye'),
                    np.column_stack([tth, I, bg, I_sub, sigma, px_cnt]),
                    header=hdr, fmt='%.6f')
            except Exception as e:
                log.error("Export I_sub failed for %s: %s", base, e, exc_info=True)

        self.lbl_progress.setText(
            f"Done — {n} file(s) saved to lineouts_sub{out_suffix}/")

    def _apply_rebin(self, result):
        """Apply rebin spinbox factor to lineout arrays in-place."""
        factor = self.spin_rebin.value()
        if factor <= 1 or 'tth' not in result or 'px_cnt' not in result:
            return
        tth_r, I_r, px_cnt_r, sigma_r = rebin_lineout(
            result['tth'], result['I'], result['px_cnt'], factor)
        bg_r = snip_background(I_r)
        result['tth']    = tth_r
        result['I']      = I_r
        result['px_cnt'] = px_cnt_r
        result['sigma']  = sigma_r
        result['bg']    = bg_r
        result['I_sub'] = I_r - bg_r

    def _apply_snip_range(self, result):
        """Restrict SNIP background to the user-defined 2θ range (display only)."""
        if 'tth' not in result or 'I' not in result:
            return
        lo = self.display.tth_snip_lo
        hi = self.display.tth_snip_hi
        tth = result['tth']
        I   = result['I']
        # Only restrict if the range differs meaningfully from the full coverage
        full_lo = self.geom.get('tth_min', 0.0) if self.geom else 0.0
        full_hi = self.geom.get('tth_max', 180.0) if self.geom else 180.0
        if abs(lo - full_lo) < 0.005 and abs(hi - full_hi) < 0.005:
            return
        in_range = (tth >= lo) & (tth <= hi)
        if not in_range.any():
            return
        bg = np.full_like(I, np.nan)
        bg[in_range] = snip_background(I[in_range])
        result['bg']    = bg
        result['I_sub'] = I - bg  # NaN outside the SNIP range

    def _on_tth_snip_changed(self, lo, hi):
        """Called when SNIP vlines are dragged — recompute SNIP for all displayed frames."""
        self._recompute_current_lineout()
        self._recompute_all_overlays()

    def _recompute_current_lineout(self):
        """Recompute lineout for the current image with the current mask state.

        No disk write — display only.
        """
        if (self._current_image is None or
                self.tth_lut is None or self.geom is None):
            return
        eff_mask = (self.mask if (self.mask is not None and self._use_mask)
                    else np.zeros(self._current_image.shape, dtype=bool))
        try:
            result = reduce_frame(
                self._current_image, eff_mask,
                self.tth_lut, self.eta_lut, self.geom,
                bin_maps=self.bin_maps,
                eta_min=self.spin_eta_min.value(),
                eta_max=self.spin_eta_max.value())
            self._apply_rebin(result)
            self._apply_snip_range(result)
            result['image'] = self._current_image
            self.btn_export_isub.setEnabled(True)
            self.display.show_results(
                result, title=self._current_base or 'current')
        except Exception as e:
            log.error("Recompute failed for %s: %s", self._current_base or '?', e, exc_info=True)
            self.lbl_progress.setText(f"Recompute error: {e}")

    def _recompute_all_overlays(self):
        """Recompute lineouts for every checked overlay file with the current
        eta and SNIP range.  Only files whose raw TIFF exists in data_folder
        are reprocessed; disk-only files are left unchanged.  No disk write.
        """
        if (self.data_folder is None or
                self.tth_lut is None or self.geom is None):
            return
        cur = self._current_base
        for base in list(self._overlay_basenames):
            if base == cur:
                continue   # current file is handled by _recompute_current_lineout
            tif_path = self.data_folder / f'{base}.tif'
            if not tif_path.is_file():
                continue   # no raw TIFF — cannot reintegrate; leave stale lineout
            try:
                image = load_tiff(tif_path).astype(np.float64)
                eff_mask = (self.mask if (self.mask is not None and self._use_mask)
                            else np.zeros(image.shape, dtype=bool))
                result = reduce_frame(
                    image, eff_mask,
                    self.tth_lut, self.eta_lut, self.geom,
                    bin_maps=self.bin_maps,
                    eta_min=self.spin_eta_min.value(),
                    eta_max=self.spin_eta_max.value())
                self._apply_rebin(result)
                self._apply_snip_range(result)
                entry = (result['tth'], result['I'],
                         result.get('bg'), result.get('I_sub'), result.get('sigma'),
                         result.get('px_cnt'))
                self._lineout_store[base] = entry
                self.display.add_lineout(base, *entry)
            except Exception as e:
                log.warning("Overlay recompute failed for %s: %s", base, e)

    def _reprocess_and_save_current(self):
        """Recompute and save the current file with the updated mask."""
        if self._current_base is None or self.data_folder is None:
            return
        tif_path = self.data_folder / f'{self._current_base}.tif'
        if tif_path.is_file():
            self._process_single(tif_path, save_output=True)

    # ── Theme / utility ──────────────────────────────────────────────────────

    def apply_mpl_theme(self, colors):
        self._mpl_colors = colors
        self.display.apply_mpl_theme(colors)

    def set_last_image_folder(self, folder):
        """Set the starting directory for Browse (called when Calib tab loads an image)."""
        if folder is not None:
            self._last_image_folder = Path(folder)
