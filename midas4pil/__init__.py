"""
midas4pil — lightweight powder/HP diffraction reduction for Pilatus/Eiger at HPCAT, APS.

Standalone calibration (no pyFAI/Dioptas required)
--------------------------------------------------
    from midas4pil.io import make_geometry, save_params, load_tiff, auto_mask
    from midas4pil.geometry import find_beam_center_auto
    from midas4pil.calibrant import load_calibrant
    from midas4pil.panels import make_panel_id_map
    from midas4pil.optimizer import calibrate

    image = load_tiff("calibrant.tif").astype("float64")
    nrows, ncols = image.shape
    # Build panel map for your detector (e.g. Pilatus 2M CdTe: 3x8 modules, 487x195 px, 7/17 px gaps)
    panel_map = make_panel_id_map(nrows, ncols, 3, 8, 487, 195, gap_y=7, gap_z=17)
    mask = auto_mask(image, panel_map)
    bc_y, bc_z = find_beam_center_auto(image, mask)
    px = 172  # pixel size in µm
    geom = make_geometry(wavelength=0.42460, lsd=350000, px=px,
                         nrows=nrows, ncols=ncols, bc_y=bc_y, bc_z=bc_z)
    rings = load_calibrant("CeO2.jcpds", geom['wavelength'], tth_max=29.0)
    result = calibrate(image, mask, geom, panel_map, rings)
    save_params(result['geom'], "geometry.toml")

Data reduction
--------------
    from midas4pil.io import load_params, load_tiff, load_mask
    from midas4pil.geometry import build_lut
    from midas4pil.integrate import reduce_frame

    geom     = load_params("geometry.toml")
    tth_lut, eta_lut = build_lut(**{k: geom[k] for k in
                   ['nrows','ncols','bc_y','bc_z','lsd','px',
                    'tx_deg','ty_deg','tz_deg','p0','p1','p2','p3','p4','rho_d']})
    image = load_tiff("frame.tif")
    mask  = load_mask("mask.tif")
    result = reduce_frame(image, mask, tth_lut, eta_lut, geom)
    # result['tth'], result['I'], result['bg'], result['I_sub'], result['sigma'], result['px_cnt']
    # result['cake_img'], result['tth_cake'], result['eta_cake'], result['px_cnt_cake']
"""

from .geometry  import (build_tilt_matrix, pixel_to_r_eta, r_to_tth, build_lut,
                        pixel_resolution, varbin_tth_edges, lut_tth_range,
                        find_beam_center, find_beam_center_auto,
                        fit_circle, lsd_from_ring)
from .integrate import (snip_background, integrate_1d, integrate_1d_varbin,
                        reduce_frame, precompute_bin_maps, rebin_lineout)
from .cake      import cake, cake_varbin
from .io        import (read_poni, write_poni, make_geometry,
                        save_params, load_params,
                        load_midas_params, save_midas_params,
                        load_tiff, load_mask,
                        orient_mask, auto_mask, save_mask)
from .calibrant import read_jcpds, read_cif, ring_table, load_calibrant
from .optimizer import calibrate
from .panels    import (make_panel_id_map,
                        read_panel_shifts, save_panel_shifts,
                        apply_panel_offsets, build_lut_with_panels)

__version__      = "1.1.3"
__release_date__ = "2026-04-27"
__author__       = "Changyong Park, HPCAT, X-ray Science Division, Argonne National Laboratory (cypark@anl.gov)"
__coauthor__     = "Claude Code (Anthropic AI)"
__license__      = "HPCAT Internal Use License"
__all__ = [
    # geometry
    "build_tilt_matrix", "pixel_to_r_eta", "r_to_tth", "build_lut",
    "pixel_resolution", "varbin_tth_edges", "lut_tth_range",
    "find_beam_center", "find_beam_center_auto",
    "fit_circle", "lsd_from_ring",
    # integration
    "snip_background", "integrate_1d", "integrate_1d_varbin",
    "reduce_frame", "precompute_bin_maps", "rebin_lineout",
    # caking
    "cake", "cake_varbin",
    # panel corrections
    "make_panel_id_map",
    "read_panel_shifts", "save_panel_shifts",
    "apply_panel_offsets", "build_lut_with_panels",
    # I/O
    "read_poni", "write_poni", "make_geometry",
    "save_params", "load_params",
    "load_midas_params", "save_midas_params",
    "load_tiff", "load_mask",
    "orient_mask", "auto_mask", "save_mask",
    # calibrant
    "read_jcpds", "read_cif", "ring_table", "load_calibrant",
    # optimizer
    "calibrate",
]
