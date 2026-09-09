# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import importlib
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
__version__ = "0.5.2+lazyimports"

__all__: list[str] = []


def _import_optional_public_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = exc.name
        # The extension package is optional. Treat its absence, or the absence
        # of an external runtime dependency, as the extension being unavailable.
        if missing == module_name or (missing is not None and missing.split('.', 1)[0] != 'fla'):
            return None
        raise


_PUBLIC_MODULES = ('fla.layers', 'fla.models')


def __getattr__(name: str):
    """Resolves layers and models on first access; importing them eagerly would pull in transformers and
    every kernel family for users who only want a kernel."""
    if name in ('layers', 'models'):
        return importlib.import_module(f'fla.{name}')
    for module_name in _PUBLIC_MODULES:
        module = _import_optional_public_module(module_name)
        if module is not None and name in module.__all__:
            return getattr(module, name)
    raise AttributeError(f"module 'fla' has no attribute {name!r}")
