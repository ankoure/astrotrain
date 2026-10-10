import ctypes
import importlib.util
from pathlib import Path


def _preload_cusolver():
    """
    TensorFlow 2.21 finds the pip-installed CUDA libraries (the `gpu` extra)
    through a list of paths built into it, but cusolver is missing from that list,
    so it can't load libcusolver.so.11 and skips the GPU entirely. Loading the
    library by its full path first means TensorFlow's later load by name finds it
    already loaded. Does nothing without the `gpu` extra.
    """
    try:
        spec = importlib.util.find_spec("nvidia.cusolver")
    except ModuleNotFoundError:  # no `nvidia` package at all
        return
    if spec is None or not spec.submodule_search_locations:
        return
    library = Path(spec.submodule_search_locations[0]) / "lib" / "libcusolver.so.11"
    if library.exists():
        ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)


# Must run before anything in the package imports TensorFlow
_preload_cusolver()
