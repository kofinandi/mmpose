"""Compatibility shims to allow importing external/sapiens2 in the MMPose
environment (Python 3.8, PyTorch 2.4) without needing sapiens2's training
dependencies.

Call ``install_sapiens2_shims(sapiens2_root)`` once **before**
``import sapiens`` runs.  All shims are idempotent.

Problems addressed
------------------
1. ``sapiens/engine/runners/base_runner.py`` imports training-only symbols
   (``accelerate``, ``torch.distributed.fsdp.MixedPrecisionPolicy`` from
   PyTorch ≥ 2.6, etc.).  We pre-stub the entire runners sub-package so its
   real source is never executed.

2. Several sapiens2 source files use the ``X | Y`` type-union syntax
   introduced in Python 3.10 (e.g. ``float | None``).  We register a custom
   meta path finder that prepends ``from __future__ import annotations`` when
   loading any sapiens source file, making annotations lazy strings so the
   ``|`` operator is never evaluated at class/function definition time.

3. ``iopath`` is stubbed for any utilities that import it.
"""

import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
import types

_shims_installed = False


def install_sapiens2_shims(sapiens2_root: str) -> None:
    """Install all compatibility shims for sapiens2.

    Must be called with ``sapiens2_root`` (the directory containing the
    ``sapiens/`` package) **before** any ``import sapiens`` statement.

    Safe to call multiple times; installation only happens once.
    """
    global _shims_installed
    if _shims_installed:
        return
    _shims_installed = True

    # ---- sapiens.engine.runners: pre-stub to skip training imports ----
    _stub_sapiens_runner()

    # ---- sapiens.dense: pre-stub to skip Python 3.10+ match syntax ---
    _stub_sapiens_dense()

    # ---- Python 3.10+ union annotation syntax fixer ------------------
    # Register a meta path finder that prepends
    # ``from __future__ import annotations`` to every sapiens .py file so
    # that ``float | None`` and similar annotations are treated as lazy
    # strings rather than evaluated expressions.
    sys.meta_path.insert(0, _FutureAnnotationsFinder(sapiens2_root))

    # ---- .safetensors checkpoint loader for MMEngine ------------------
    # MMEngine's default CheckpointLoader tries torch.load() on all local
    # files, which fails for .safetensors.  Register a regex-based scheme
    # so MMEngine uses the safetensors library for those files.
    _register_safetensors_loader()

    # ---- iopath -------------------------------------------------------
    if 'iopath' not in sys.modules:
        iopath_mod = types.ModuleType('iopath')
        iopath_common = types.ModuleType('iopath.common')
        iopath_file_io = types.ModuleType('iopath.common.file_io')

        class _PathManager:
            def open(self, path, mode='r', **kwargs):
                return open(path, mode)

            def get_local_path(self, path):
                return path

            def isfile(self, path):
                return os.path.isfile(path)

        iopath_file_io.PathManager = _PathManager
        sys.modules['iopath'] = iopath_mod
        sys.modules['iopath.common'] = iopath_common
        sys.modules['iopath.common.file_io'] = iopath_file_io
        iopath_mod.common = iopath_common
        iopath_common.file_io = iopath_file_io


# ---------------------------------------------------------------------------
# Runner stub
# ---------------------------------------------------------------------------

def _stub_sapiens_runner():
    """Pre-register stub modules for the sapiens runner sub-package.

    ``base_runner.py`` needs ``accelerate`` and FSDP symbols from PyTorch
    ≥ 2.6 that are not available in the mmpose environment.  Stubbing the
    entire sub-package prevents its source from ever being executed.
    """

    class _BaseRunner:
        """Stub for sapiens2 BaseRunner (training only; unused at inference)."""
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                'BaseRunner is a training-only class and is not available '
                'in the MMPose inference environment.')

    br_mod = types.ModuleType('sapiens.engine.runners.base_runner')
    br_mod.BaseRunner = _BaseRunner
    sys.modules['sapiens.engine.runners.base_runner'] = br_mod

    runners_mod = types.ModuleType('sapiens.engine.runners')
    runners_mod.BaseRunner = _BaseRunner
    runners_mod.base_runner = br_mod
    runners_mod.__all__ = ['BaseRunner']
    sys.modules['sapiens.engine.runners'] = runners_mod


def _stub_sapiens_dense():
    """Pre-register a stub for the sapiens.dense sub-package.

    ``sapiens/dense/src/models/losses/utils.py`` uses Python 3.10+
    ``match`` statement syntax which cannot be patched by
    ``from __future__ import annotations``.  The dense sub-package
    contains segmentation/normals/albedo models that are not needed for
    pose estimation, so we stub it out entirely.
    """
    for mod_name in [
        'sapiens.dense',
        'sapiens.dense.src',
        'sapiens.dense.src.models',
        'sapiens.dense.src.models.losses',
        'sapiens.dense.src.evaluators',
        'sapiens.dense.src.runners',
        'sapiens.dense.src.datasets',
        'sapiens.dense.src.visualizers',
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)


# ---------------------------------------------------------------------------
# Python 3.10+ type annotation fixer
# ---------------------------------------------------------------------------

class _FutureAnnotationsLoader(importlib.machinery.SourceFileLoader):
    """Source loader that prepends ``from __future__ import annotations``
    before compiling, making ``X | Y`` annotations safe on Python 3.8/3.9.
    """

    def get_code(self, fullname):
        source = self.get_source(fullname)
        if source is None:
            return None
        # Inject future import at the top of the source so that all
        # annotations are treated as strings (lazy) and never evaluated.
        patched = 'from __future__ import annotations\n' + source
        filename = self.get_filename(fullname)
        return compile(patched, filename, 'exec', dont_inherit=True)


class _FutureAnnotationsFinder(importlib.abc.MetaPathFinder):
    """Meta path finder that intercepts sapiens module imports and uses
    ``_FutureAnnotationsLoader`` to handle Python 3.10+ annotation syntax.
    """

    def __init__(self, sapiens2_root: str):
        self._root = sapiens2_root

    def find_spec(self, fullname, path, target=None):
        # Only intercept sapiens modules
        if not fullname.startswith('sapiens'):
            return None
        # Already in sys.modules (e.g. our stubs) – leave them alone
        if fullname in sys.modules:
            return None

        search_paths = list(path) if path else [
            os.path.join(self._root, 'sapiens')
        ]

        parts = fullname.split('.')
        # For sub-modules, just look in the provided search path
        name = parts[-1]

        for search_path in search_paths:
            pkg_dir = os.path.join(search_path, name)
            init_file = os.path.join(pkg_dir, '__init__.py')
            mod_file = os.path.join(search_path, name + '.py')

            if os.path.isfile(init_file):
                loader = _FutureAnnotationsLoader(fullname, init_file)
                spec = importlib.util.spec_from_file_location(
                    fullname,
                    init_file,
                    loader=loader,
                    submodule_search_locations=[pkg_dir],
                )
                return spec

            if os.path.isfile(mod_file):
                loader = _FutureAnnotationsLoader(fullname, mod_file)
                spec = importlib.util.spec_from_file_location(
                    fullname,
                    mod_file,
                    loader=loader,
                )
                return spec

        return None


# ---------------------------------------------------------------------------
# safetensors checkpoint loader
# ---------------------------------------------------------------------------

def _register_safetensors_loader():
    """Register a MMEngine ``CheckpointLoader`` backend for ``.safetensors``
    files.

    MMEngine dispatches to ``load_from_local`` (``torch.load``) for any path
    that doesn't match a known URL scheme.  ``torch.load`` cannot read
    ``.safetensors`` format.  This function registers a regex scheme that
    matches paths ending in ``.safetensors`` and loads them with the
    ``safetensors`` library instead.

    The returned dict has a ``state_dict`` key so that MMEngine's
    ``_load_checkpoint_to_model`` handles it the same way as a regular
    PyTorch checkpoint.
    """
    from mmengine.runner.checkpoint import CheckpointLoader
    from safetensors.torch import load_file

    def _load_safetensors(filename, map_location=None):
        device = 'cpu'
        if map_location is not None:
            device = str(map_location)
        tensors = load_file(filename, device=device)
        return {'state_dict': tensors}

    _load_safetensors.__name__ = 'load_from_safetensors'

    CheckpointLoader.register_scheme(
        prefixes=r'.+\.safetensors$',
        loader=_load_safetensors,
        force=True,
    )
