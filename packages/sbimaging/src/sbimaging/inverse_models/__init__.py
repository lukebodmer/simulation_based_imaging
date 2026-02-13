"""Inverse models for simulation-based imaging.

This module provides inverse models for learning mappings from
sensor data to material properties or k-space coefficients.

Available models:
- NeuralNetworkModel: CNN or MLP architecture
- GaussianProcessModel: Parallel Partial GP emulation via PyRobustGaSP
"""

from sbimaging.inverse_models.base import DataLoader, InverseModel
from sbimaging.inverse_models.gp import GaussianProcessModel, PyRobustGaSP
from sbimaging.inverse_models.nn import NeuralNetworkModel

__all__ = [
    "DataLoader",
    "GaussianProcessModel",
    "InverseModel",
    "NeuralNetworkModel",
    "PyRobustGaSP",
]
