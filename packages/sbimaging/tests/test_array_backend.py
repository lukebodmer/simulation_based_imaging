"""Tests for array backend abstraction."""

import numpy as np

from sbimaging.array import get_array_module, xp
from sbimaging.array.backend import GPU_AVAILABLE, to_gpu, to_numpy


def test_xp_is_valid_module():
    """xp should be either numpy or cupy."""
    assert hasattr(xp, "zeros")
    assert hasattr(xp, "array")
    assert hasattr(xp, "ndarray")


def test_get_array_module_returns_module():
    """get_array_module should return a valid array module."""
    module = get_array_module()
    assert hasattr(module, "zeros")


def test_to_numpy_from_numpy():
    """to_numpy should handle numpy arrays."""
    arr = np.array([1, 2, 3])
    result = to_numpy(arr)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, arr)


def test_to_gpu_returns_array():
    """to_gpu should return an array (GPU or CPU depending on availability)."""
    arr = np.array([1, 2, 3])
    result = to_gpu(arr)
    assert hasattr(result, "shape")
    assert result.shape == (3,)


def test_roundtrip_conversion():
    """Converting to GPU and back should preserve values."""
    original = np.array([1.0, 2.0, 3.0])
    gpu_arr = to_gpu(original)
    back = to_numpy(gpu_arr)
    assert np.allclose(back, original)


def test_gpu_available_is_bool():
    """GPU_AVAILABLE should be a boolean."""
    assert isinstance(GPU_AVAILABLE, bool)
