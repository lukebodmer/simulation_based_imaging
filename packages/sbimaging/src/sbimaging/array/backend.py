"""GPU/CPU array backend detection and abstraction."""

import numpy as np

try:
    import cupy as cp

    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False


def get_array_module(arr=None):
    """Return the array module for the given array.

    If no array is provided, returns the default backend (cupy if available).

    Args:
        arr: Optional array to detect module from.

    Returns:
        cupy or numpy module.
    """
    if arr is not None and GPU_AVAILABLE:
        return cp.get_array_module(arr)
    return cp if GPU_AVAILABLE else np


def to_numpy(arr):
    """Convert array to numpy, regardless of backend.

    Args:
        arr: Array (numpy or cupy).

    Returns:
        numpy.ndarray.
    """
    if GPU_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def to_gpu(arr):
    """Convert array to GPU if available, otherwise return as-is.

    Args:
        arr: Array (numpy or cupy).

    Returns:
        cupy.ndarray if GPU available, else numpy.ndarray.
    """
    if GPU_AVAILABLE:
        return cp.asarray(arr)
    return np.asarray(arr)


# Default array module
xp = get_array_module()
