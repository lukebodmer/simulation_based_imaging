"""Gaussian process inverse models.

This module wraps PyRobustGaSP for Parallel Partial Emulation.
"""

from sbimaging.inverse_models.gp.PyRobustGaSP import PyRobustGaSP
from sbimaging.inverse_models.gp.emulator import GaussianProcessModel

__all__ = ["GaussianProcessModel", "PyRobustGaSP"]
