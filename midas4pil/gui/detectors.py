"""
Detector preset database.

Each entry provides nrows, ncols, pixel size, and optional panel layout
for tiled detectors (module dimensions and gap widths).
"""

# Panel layout: (n_panels_y, n_panels_z, panel_size_y, panel_size_z, gap_y, gap_z)
# gap_y and gap_z are per-gap widths (list or scalar).

DETECTORS = {
    # ── Dectris Pilatus (172 µm) ──
    "Pilatus 100K": {
        "nrows": 195, "ncols": 487, "px_um": 172,
    },
    "Pilatus 200K": {
        "nrows": 407, "ncols": 487, "px_um": 172,
        "panels": (1, 2, 487, 195, 0, 17),
    },
    "Pilatus 300K": {
        "nrows": 619, "ncols": 487, "px_um": 172,
        "panels": (1, 3, 487, 195, 0, 17),
    },
    "Pilatus 1M": {
        "nrows": 1043, "ncols": 981, "px_um": 172,
        "panels": (2, 5, 487, 195, 7, 17),
    },
    "Pilatus 2M CdTe": {
        "nrows": 1679, "ncols": 1475, "px_um": 172,
        "panels": (3, 8, 487, 195, 7, 17),
    },
    "Pilatus 6M": {
        "nrows": 2527, "ncols": 2463, "px_um": 172,
        "panels": (5, 12, 487, 195, 7, 17),
    },

    # ── Dectris Eiger2 (75 µm) ──
    "Eiger2 X 1M": {
        "nrows": 1065, "ncols": 1030, "px_um": 75,
        "panels": (2, 1, 514, 1030, 37, 0),
    },
    "Eiger2 X 4M": {
        "nrows": 2167, "ncols": 2070, "px_um": 75,
        "panels": (2, 4, 514, 1030, 37, 10),
    },
    "Eiger2 X 9M": {
        "nrows": 3269, "ncols": 3110, "px_um": 75,
        "panels": (3, 6, 514, 1030, 37, 10),
    },
    "Eiger2 X 16M": {
        "nrows": 4371, "ncols": 4150, "px_um": 75,
        "panels": (4, 8, 514, 1030, 37, 10),
    },

    # ── Flat panels (200 µm) ──
    "Perkin-Elmer XRD 1621": {
        "nrows": 2048, "ncols": 2048, "px_um": 200,
    },
    "Perkin-Elmer XRD 0822": {
        "nrows": 1024, "ncols": 1024, "px_um": 200,
    },
    "GE-41RT / Varex 4343CT": {
        "nrows": 2048, "ncols": 2048, "px_um": 200,
    },

    # ── X-Spectrum Lambda (55 µm) ──
    "Lambda 750K": {
        "nrows": 516, "ncols": 1556, "px_um": 55,
    },
    "Lambda 2M": {
        "nrows": 1556, "ncols": 1556, "px_um": 55,
    },

    # ── Rayonix ──
    "Rayonix MX225-HS": {
        "nrows": 3072, "ncols": 3072, "px_um": 73.2,
    },
    "Rayonix MX300-HS": {
        "nrows": 4096, "ncols": 4096, "px_um": 73.2,
    },
    "Rayonix SX165": {
        "nrows": 2048, "ncols": 2048, "px_um": 79.0,
    },

    # ── Custom ──
    "Custom": {
        "nrows": 2048, "ncols": 2048, "px_um": 172,
    },
}

def make_panel_map(detector_name):
    """Return a panel-ID map for the named detector, or None for monolithic detectors."""
    from ..panels import make_panel_id_map
    spec = DETECTORS.get(detector_name)
    if spec is None:
        raise ValueError(f"Unknown detector: {detector_name!r}")
    panels = spec.get("panels")
    if panels is None:
        return None
    n_panels_y, n_panels_z, panel_size_y, panel_size_z, gap_y, gap_z = panels
    return make_panel_id_map(
        nrows=spec["nrows"], ncols=spec["ncols"],
        n_panels_y=n_panels_y, n_panels_z=n_panels_z,
        panel_size_y=panel_size_y, panel_size_z=panel_size_z,
        gap_y=gap_y, gap_z=gap_z,
    )


def make_panel_map_from_shape(nrows, ncols):
    """Return a panel-ID map by matching image shape to a detector preset, or None."""
    from ..panels import make_panel_id_map
    for spec in DETECTORS.values():
        if spec["nrows"] == nrows and spec["ncols"] == ncols and "panels" in spec:
            n_panels_y, n_panels_z, panel_size_y, panel_size_z, gap_y, gap_z = spec["panels"]
            return make_panel_id_map(
                nrows=nrows, ncols=ncols,
                n_panels_y=n_panels_y, n_panels_z=n_panels_z,
                panel_size_y=panel_size_y, panel_size_z=panel_size_z,
                gap_y=gap_y, gap_z=gap_z,
            )
    return None


# Display order: group by family, default first
DETECTOR_NAMES = [
    "Pilatus 2M CdTe",
    "Pilatus 100K", "Pilatus 200K", "Pilatus 300K",
    "Pilatus 1M", "Pilatus 6M",
    "Eiger2 X 1M", "Eiger2 X 4M", "Eiger2 X 9M", "Eiger2 X 16M",
    "Perkin-Elmer XRD 1621", "Perkin-Elmer XRD 0822",
    "GE-41RT / Varex 4343CT",
    "Lambda 750K", "Lambda 2M",
    "Rayonix MX225-HS", "Rayonix MX300-HS", "Rayonix SX165",
    "Custom",
]

DEFAULT_DETECTOR = "Pilatus 2M CdTe"
