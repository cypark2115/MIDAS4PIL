# Copyright (c) 2026, UChicago Argonne, LLC. All Rights Reserved.
# Author: Changyong Park, HPCAT, X-ray Science Division, Argonne National Laboratory.
# BSD Open Source License -- see LICENSE in the project root for the full license text.
"""Session logging for midas4pil.

Log files are written to:
  - <cwd>/logs/  when running from the source tree (dev mode)
  - ~/.midas4pil/logs/  for an installed/distributed package

All midas4pil loggers (named 'midas4pil.*') write to the file at INFO+
and echo WARNING+ to stderr.

Usage
-----
    from midas4pil.gui.logging_setup import setup_logging
    log_path = setup_logging()
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

_FMT_FILE    = '%(asctime)s [%(levelname)-7s] %(name)s: %(message)s'
_FMT_CONSOLE = '[%(levelname)s] %(name)s: %(message)s'
_DATEFMT     = '%Y-%m-%d %H:%M:%S'


def _default_log_dir() -> Path:
    # Source tree: pyproject.toml is present in cwd
    if (Path.cwd() / 'pyproject.toml').exists():
        return Path.cwd() / 'logs'
    return Path.home() / '.midas4pil' / 'logs'


def setup_logging(log_dir=None):
    """Configure file + console logging for one midas4pil session.

    Parameters
    ----------
    log_dir : str or Path, optional
        Directory for log files.  Defaults to ``<cwd>/logs/`` in the source
        tree, or ``~/.midas4pil/logs/`` for an installed package.

    Returns
    -------
    Path
        Absolute path to the session log file.
    """
    log_dir = Path(log_dir) if log_dir else _default_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file  = log_dir / f'midas4pil_{timestamp}.log'

    root = logging.getLogger('midas4pil')
    root.setLevel(logging.DEBUG)
    root.propagate = False   # don't double-emit to Python root logger

    # Clear any handlers from a previous call (e.g. during testing)
    root.handlers.clear()

    # File handler — INFO and above
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(_FMT_FILE, datefmt=_DATEFMT))
    root.addHandler(fh)

    # Console handler — WARNING and above
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter(_FMT_CONSOLE))
    root.addHandler(ch)

    root.info("Session started — log: %s", log_file)
    return log_file
