# Copyright (c) 2026, UChicago Argonne, LLC. All Rights Reserved.
# Author: Changyong Park, HPCAT, X-ray Science Division, Argonne National Laboratory.
# BSD Open Source License -- see LICENSE in the project root for the full license text.
"""
SyncBus — shared signal bus for synchronized 2theta cursor across all canvases.

One instance is created in MainWindow and passed to CakeLineoutWidget and
StackWidget.  Any canvas that detects a 2theta position emits tth_changed;
all other canvases listen and draw their respective indicators.
"""

from PySide6.QtCore import QObject, Signal


class SyncBus(QObject):
    """Lightweight signal bus for cross-widget 2theta synchronization."""

    # Emitted by whichever canvas currently has the mouse.
    # Receivers draw a cursor/line at this 2theta value.
    tth_changed = Signal(float)

    # Emitted when the mouse leaves all canvases — receivers hide their cursor.
    clear_hover = Signal()
