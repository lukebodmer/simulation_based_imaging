"""2D inverse models for FDTD simulations.

Provides k-space representation and data loading utilities for
training inverse models on 2D acoustic wave simulation data.
"""

from sbimaging.inverse_models.dim2.kspace import (
    KSpace2D,
    create_inclusion_image,
    inclusion_to_kspace,
    kspace_to_image,
)
from sbimaging.inverse_models.dim2.data import DataLoader2D, prepare_training_data
from sbimaging.inverse_models.dim2.network import NeuralNetwork2D
from sbimaging.inverse_models.dim2.train import (
    predict_and_visualize,
    train_2d_inverse_model,
)

__all__ = [
    "DataLoader2D",
    "KSpace2D",
    "NeuralNetwork2D",
    "create_inclusion_image",
    "inclusion_to_kspace",
    "kspace_to_image",
    "predict_and_visualize",
    "prepare_training_data",
    "train_2d_inverse_model",
]
