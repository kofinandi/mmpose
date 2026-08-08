"""Compatibility shims required to import ``external/AlphaPose`` unmodified.

AlphaPose's tracking code was written against NumPy < 1.20, where
``np.float`` / ``np.int`` / ``np.bool`` were aliases for the Python
builtins.  Those aliases were deprecated in 1.20 and **removed** in 1.24
(this repo pins 1.24.3), so several upstream modules raise
``AttributeError`` on import or on the first frame:

* ``trackers/tracker_api.py:38`` -- ``np.asarray(tlwh, dtype=np.float)``
  in ``STrack.__init__``, i.e. every detection of every frame.
* ``trackers/tracking/matching.py:64,69,70,104,107`` -- the IoU and
  embedding cost matrices.
* ``trackers/utils/bbox.py``, ``trackers/ReidModels/*`` -- misc.
* ``cython_bbox`` itself (``bbox_overlaps``, imported by ``matching.py``)
  defines ``DTYPE = np.float`` at module scope.

Rather than patch the submodule -- which would defeat the point of
vendoring it at a pinned SHA and make upstream diffs unreadable -- restore
the aliases before importing.  They are exact aliases for the builtins, so
this reinstates 1.19 semantics rather than approximating them.

Unlike ``petr_compat``/``pavenet_compat``, there is no ``finalize`` step:
the aliases are needed at *inference* time (``STrack.__init__`` runs per
detection per frame), not only at import time, so they must stay installed
for the lifetime of the process.  Only the three names upstream actually
uses are restored, to keep the blast radius as small as possible; in
particular ``np.object``/``np.str`` are left alone.
"""

import os
import sys

import numpy as np

# The builtin each removed alias stood for, per the NumPy 1.20 deprecation
# notes (https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations).
_LEGACY_NUMPY_ALIASES = {
    'float': float,
    'int': int,
    'bool': bool,
}

_shims_installed = False


def install_alphapose_shims() -> None:
    """Restore the removed NumPy builtin aliases (idempotent)."""
    global _shims_installed
    if _shims_installed:
        return
    for name, builtin in _LEGACY_NUMPY_ALIASES.items():
        if not hasattr(np, name):
            setattr(np, name, builtin)
    _shims_installed = True


def ensure_alphapose_on_path(alphapose_root: str) -> str:
    """Put ``alphapose_root`` and its ``trackers/`` package dir on sys.path.

    Two entries are needed because ``trackers/tracker_api.py`` mixes import
    styles: it does ``from trackers.utils import kalman_filter`` (needs the
    repo root) *and* ``from utils.utils import *`` after inserting its own
    directory (needs ``<root>/trackers``).  Inserting both up front keeps
    the import order deterministic instead of depending on upstream's
    ``sys.path.insert`` side effect having run first.

    Returns the absolute root path.
    """
    root = os.path.abspath(alphapose_root)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f'AlphaPose repository not found at {root!r}. It is vendored as '
            f'a git submodule; run '
            f'`git submodule update --init external/AlphaPose`, then build '
            f'its CUDA extensions with '
            f'`pip install -e external/AlphaPose --no-build-isolation '
            f'--no-deps`.')
    for entry in (os.path.join(root, 'trackers'), root):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return root
