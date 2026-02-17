"""1D inverse models for FDTD simulations.

This module provides neural network models for learning inverse mappings
from 1D FDTD sensor data to physical density profiles.
"""

from sbimaging.inverse_models.dim1.data import DataLoader1D, prepare_training_data
from sbimaging.inverse_models.dim1.network import NeuralNetwork1D
from sbimaging.inverse_models.dim1.train import train_1d_inverse_model

__all__ = [
    "DataLoader1D",
    "prepare_training_data",
    "NeuralNetwork1D",
    "train_1d_inverse_model",
]
