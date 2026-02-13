"""Array backend abstraction for GPU/CPU computation.

Provides a unified interface using CuPy when available, falling back to NumPy.

Usage:
    from sbimaging.array import xp

    # xp is either cupy or numpy
    arr = xp.zeros((10, 10))
"""

from sbimaging.array.backend import get_array_module, xp

__all__ = ["xp", "get_array_module"]
