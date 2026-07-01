# Copyright (c) 2026, UChicago Argonne, LLC. All Rights Reserved.
# Author: Changyong Park, HPCAT, X-ray Science Division, Argonne National Laboratory.
# BSD Open Source License -- see LICENSE in the project root for the full license text.
"""
midas4pil GUI — real-time diffraction data reduction for HPCAT beamlines.

Launch:  python -m midas4pil.gui
"""


def launch_gui(argv=None):
    """Launch the midas4pil GUI application."""
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        raise ImportError(
            "PySide6 is required for the GUI.\n"
            "Install with:  pip install --user PySide6 matplotlib>=3.5\n"
            "  or:          pip install --user midas4pil[gui]"
        )

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        raise ImportError(
            "matplotlib is required for the GUI.\n"
            "Install with:  pip install --user matplotlib>=3.5"
        )

    import sys
    from .logging_setup import setup_logging

    log_path = setup_logging()   # must happen before any module imports that create loggers

    from .main_window import MainWindow

    app = QApplication(argv if argv is not None else sys.argv)
    app.setStyle('Fusion')   # palette-driven on all platforms; avoids native-widget override on Windows
    app.setApplicationName("midas4pil")

    window = MainWindow()
    window.statusBar().showMessage(f"Ready — log: {log_path}")
    window.show()
    sys.exit(app.exec())
