"""
Batch processing worker thread.

Runs reduce_frame() in a background QThread so the GUI stays responsive.
Emits signals for progress updates and completed frames.
"""

import logging
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal

from ..io import load_tiff, auto_mask
from ..integrate import reduce_frame
import tifffile

log = logging.getLogger(__name__)


class BatchWorker(QThread):
    """Process .tif images in a folder using reduce_frame().

    Signals
    -------
    frame_processed(dict, str)
        Emitted after each frame: (result dict, basename).
    progress(int, int)
        Emitted after each frame: (current_index, total_count).
    batch_finished(int)
        Emitted when all files are done: (total processed count).
    error(str)
        Emitted on per-file errors (processing continues).
    """

    frame_processed = Signal(dict, str)
    progress = Signal(int, int)
    batch_finished = Signal(int)
    error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_folder = None
        self.mask = None
        self.tth_lut = None
        self.eta_lut = None
        self.geom = None
        self.bin_maps = None
        self._out_suffix = ''
        self._stop = False
        self._file_list = []
        self.eta_min = -180.0
        self.eta_max = 180.0

    def setup(self, data_folder, mask, tth_lut, eta_lut, geom, bin_maps=None,
              out_suffix='', eta_min=-180.0, eta_max=180.0):
        """Configure the worker before starting.

        Parameters
        ----------
        out_suffix : str
            Folder name suffix for lineout output.  '' → lineouts/,
            '_nomask' → lineouts_nomask/.  When '_nomask', no cake .tif
            is written (cake is not affected by the mask).
        """
        self.data_folder = Path(data_folder)
        self.mask = mask
        self.tth_lut = tth_lut
        self.eta_lut = eta_lut
        self.geom = geom
        self.bin_maps = bin_maps
        self._out_suffix = out_suffix
        self._stop = False
        self.eta_min = eta_min
        self.eta_max = eta_max

    def set_file_list(self, files):
        """Set specific files to process (for watch mode incremental)."""
        self._file_list = [Path(f) for f in files]

    def request_stop(self):
        """Signal the worker to stop after the current frame finishes."""
        self._stop = True

    def run(self):
        """Process all .tif files in data_folder and emit progress signals.

        Called automatically by QThread.start().  Do not call directly.
        """
        if self.data_folder is None:
            return

        if self._file_list:
            tif_files = sorted(self._file_list)
        else:
            tif_files = sorted(self.data_folder.glob('*.tif'))

        if not tif_files:
            self.batch_finished.emit(0)
            return

        lineout_dir = self.data_folder / f'lineouts{self._out_suffix}'
        cake_dir = self.data_folder / 'cakes'
        lineout_dir.mkdir(exist_ok=True)
        save_cake = (self._out_suffix == '')
        if save_cake:
            cake_dir.mkdir(exist_ok=True)

        processed = 0
        total = len(tif_files)

        for i, tif_path in enumerate(tif_files):
            if self._stop:
                break

            base = tif_path.stem

            lineout_path = lineout_dir / f'{base}.xye'
            cake_path = cake_dir / f'{base}.tif'
            already_done = lineout_path.exists() and (
                not save_cake or cake_path.exists())
            if already_done:
                self.progress.emit(i + 1, total)
                continue

            try:
                image = load_tiff(tif_path).astype(np.float64)

                if self.mask is None:
                    from .detectors import make_panel_map_from_shape
                    panel_map = make_panel_map_from_shape(*image.shape)
                    if panel_map is not None:
                        self.mask = auto_mask(image, panel_map=panel_map)
                        self.mask = self.mask | (panel_map == 0)
                    else:
                        self.mask = auto_mask(image)

                result = reduce_frame(
                    image, self.mask,
                    self.tth_lut, self.eta_lut, self.geom,
                    bin_maps=self.bin_maps,
                    eta_min=self.eta_min, eta_max=self.eta_max)
                result['image'] = image   # for TIFF display + mask editor

                sigma  = result.get('sigma',  np.full_like(result['I'], np.nan))
                px_cnt = result.get('px_cnt', np.full(len(result['I']), -1, dtype=int))
                wl = self.geom.get('wavelength', float('nan'))
                e_kev = 12.3984193 / wl if wl else float('nan')
                np.savetxt(
                    str(lineout_path),
                    np.column_stack([result['tth'], result['I'], sigma, px_cnt]),
                    header=(f'col1=2theta_deg  col2=I  col3=sigma_I  col4=px_cnt'
                            f'  [wavelength={wl:.7f}A  energy={e_kev:.4f}keV'
                            f'  eta={self.eta_min:.1f}-{self.eta_max:.1f}deg]'),
                    fmt='%.6f')

                if save_cake:
                    tifffile.imwrite(
                        str(cake_path),
                        result['cake_img'].T.astype(np.float32))
                    if 'tth_cake' in result and 'eta_cake' in result:
                        np.savez(str(cake_dir / f'{base}_axes.npz'),
                                 tth=result['tth_cake'],
                                 eta=result['eta_cake'])

                self.frame_processed.emit(result, base)
                processed += 1

            except Exception as e:
                log.error("Frame processing failed for %s: %s", base, e, exc_info=True)
                self.error.emit(f"{base}: {e}")

            self.progress.emit(i + 1, total)

        self.batch_finished.emit(processed)
